import random
import numpy as np
import pandas as pd

from evaluator import get_metrics

import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import structured_negative_sampling

from sklearn.model_selection import train_test_split

# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_u, num_v, embedding_dim=64, K=3, add_self_loops=False, pretrain_embs=None):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_u, self.num_v = num_u, num_v
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_u, embedding_dim=self.embedding_dim) # e_u^0
        nn.init.normal_(self.users_emb.weight, std=0.1)
        
        if pretrain_embs:
            self.items_emb = pretrain_embs
        else:
            self.items_emb = nn.Embedding(num_embeddings=self.num_v, embedding_dim=self.embedding_dim) # e_i^0
            nn.init.normal_(self.items_emb.weight, std=0.1)


    def forward(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)
                
                
class LightGCNEngine(object):
    
    def __init__(self, ent2id, lr=1e-4, device=torch.device('cpu'), pretrain_embs=None):
        
        self.device = device
        
        
        self.ent2id     = ent2id
        
        if pretrain_embs:
            self.load_data(all_embeds=pretrain_embs)
            self.model = LightGCN(num_u=self.num_u, num_v=self.num_v, pretrain_embs=self.item_embeds).to(device)
        else:
            self.load_data()
            self.model = LightGCN(num_u=self.num_u, num_v=self.num_v).to(device)
        
        self.optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler  = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
    def load_data(self, all_embeds=None):
        
        df = pd.read_csv('data/graph.csv')
        
        interaction_df = df[df['relation'] == 'uses']
        num_interactions = len(interaction_df)
        
        if all_embeds:
            targets = interaction_df['target'].unique()
            tgt_indices = [self.ent2id[tgt] for tgt in targets]
            tgt_indices.sort()
            self.item_embeds = all_embeds[tgt_indices]
        
        all_indices = [i for i in range(num_interactions)]

        train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=1)
        val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=1)
        
        for cust in interaction_df['source'].unique():
            self.ent2id[cust] = max(self.ent2id.values()) + 1
        
        edge_index = [[], []]
        for _, row in interaction_df.iterrows():
            src = row['source']
            tgt = row['target']
            edge_index[0].append(self.ent2id[src])
            edge_index[1].append(self.ent2id[tgt])
            
        self.edge_index = edge_index
                
        self.train_edge_index = edge_index[:, train_indices]
        self.val_edge_index = edge_index[:, val_indices]
        self.test_edge_index = edge_index[:, test_indices]
        
        self.num_u = interaction_df['source'].nunique()
        self.num_v = interaction_df['target'].nunique()
        
        self.train_sparse_edge_index = SparseTensor(row=self.train_edge_index[0], col=self.train_edge_index[1], sparse_sizes=(
            self.num_u + self.num_v, self.num_u + self.num_v))
        self.val_sparse_edge_index = SparseTensor(row=self.val_edge_index[0], col=self.val_edge_index[1], sparse_sizes=(
            self.num_u + self.num_v, self.num_u + self.num_v))
        self.test_sparse_edge_index = SparseTensor(row=self.test_edge_index[0], col=self.test_edge_index[1], sparse_sizes=(
            self.num_u + self.num_v, self.num_u + self.num_v))
        
    def sample_mini_batch(self, batch_size):
        """Randomly samples indices of a minibatch given an adjacency matrix

        Args:
            batch_size (int): minibatch size

        Returns:
            tuple: user indices, positive item indices, negative item indices
        """
        edges = structured_negative_sampling(self.train_edge_index)
        edges = torch.stack(edges, dim=0)
        indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
        batch = edges[:, indices]
        user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
        return user_indices, pos_item_indices, neg_item_indices    
    
    def bpr_loss(self, users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
        """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

        Args:
            users_emb_final (torch.Tensor): e_u_k
            users_emb_0 (torch.Tensor): e_u_0
            pos_items_emb_final (torch.Tensor): positive e_i_k
            pos_items_emb_0 (torch.Tensor): positive e_i_0
            neg_items_emb_final (torch.Tensor): negative e_i_k
            neg_items_emb_0 (torch.Tensor): negative e_i_0
            lambda_val (float): lambda value for regularization loss term

        Returns:
            torch.Tensor: scalar bpr loss value
        """
        reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                                pos_items_emb_0.norm(2).pow(2) +
                                neg_items_emb_0.norm(2).pow(2)) # L2 loss

        pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
        pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
        neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
        neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

        loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

        return loss
    
    def evaluation(self, exclude_edge_indices, k, lambda_val):
        """Evaluates model loss and metrics including recall, precision, ndcg @ k

        Args:
            model (LighGCN): lightgcn model
            edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
            sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
            exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
            k (int): determines the top k items to compute metrics on
            lambda_val (float): determines lambda for bpr loss

        Returns:
            tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
        """
        # get embeddings
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.model.forward(self.val_sparse_edge_index)
        edges = structured_negative_sampling(self.val_edge_index, contains_neg_self_loops=False)
        
        user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

        loss = self.bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                        neg_items_emb_final, neg_items_emb_0, lambda_val).item()

        recall, precision, ndcg = get_metrics(self.model, self.val_edge_index, exclude_edge_indices, k)

        return loss, recall, precision, ndcg
    
    def fit(self, iterations=500, batch_size=128, lamb = 1e-6, iters_per_eval=20, items_per_lr_decay=20, K=20):
        
        train_losses = []
        val_losses = []
        
        for iter in range(iterations):
            # forward propagation
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.model.forward(self.train_sparse_edge_index)

            # mini batching
            user_indices, pos_item_indices, neg_item_indices = self.sample_mini_batch(batch_size)
            user_indices, pos_item_indices, neg_item_indices = user_indices.to(self.device), pos_item_indices.to(self.device), neg_item_indices.to(self.device)
            
            users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
            
            pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

            # loss computation
            train_loss = self.bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lamb)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if iter % iters_per_eval == 0:
                self.model.eval()
                val_loss, recall, precision, ndcg = self.evaluation([self.train_edge_index], K, lamb)
                print(f"[Iteration {iter}/{iterations}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                self.model.train()

            if iter % items_per_lr_decay == 0 and iter != 0:
                self.scheduler.step()