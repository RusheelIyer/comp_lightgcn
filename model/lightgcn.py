import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluator import get_metrics_list, get_user_positive_items, print_results, plot_train_val, save_results

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
        self.items_emb = nn.Embedding(num_embeddings=self.num_v, embedding_dim=self.embedding_dim) # e_i^0
        nn.init.normal_(self.users_emb.weight, std=0.1)
        
        if pretrain_embs is not None:
            self.items_emb.weight.data = pretrain_embs
        else:
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

        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_u, self.num_v]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)
                
                
class LightGCNEngine(object):
    
    def __init__(self, params, lr=1e-3, device=torch.device('cpu'), pretrain_embs=None, ent2id=None):
        
        self.device = device
        self.ent2id     = ent2id
        self.p = params
        
        if pretrain_embs is not None:
            self.load_data(all_embeds=pretrain_embs)
            self.model = LightGCN(num_u=self.num_u, num_v=self.num_v, pretrain_embs=self.item_embeds, embedding_dim=self.item_embeds.shape[1]).to(device)
        else:
            self.load_data()
            self.model = LightGCN(num_u=self.num_u, num_v=self.num_v).to(device)
        
        self.optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler  = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
    def load_data(self, all_embeds=None):
        
        df = pd.read_csv('data/graph.csv')
        
        self.interaction_df = df[df['relation'] == 'uses']
        num_interactions = len(self.interaction_df)
        
        if all_embeds is not None:
            targets = self.interaction_df['target'].unique()
            tgt_indices = [self.ent2id[tgt] for tgt in targets]
            tgt_indices.sort()
            self.item_embeds = all_embeds[tgt_indices]
            
            for cust in self.interaction_df['source'].unique():
                self.ent2id[cust] = max(self.ent2id.values()) + 1
        
        all_indices = [i for i in range(num_interactions)]

        train_indices, test_indices = train_test_split(all_indices, test_size=0.15, random_state=42)
        val_indices, test_indices = train_test_split(test_indices, test_size=0.33, random_state=42)
        
        self.user_mapping = {user: i for i, user in enumerate(self.interaction_df['source'].unique())}
        self.item_mapping = {item: i for i, item in enumerate(self.interaction_df['target'].unique())}
        
        edge_index = [[], []]
        for _, row in self.interaction_df.iterrows():
            src = row['source']
            tgt = row['target']
            # edge_index[0].append(self.ent2id[src])
            # edge_index[1].append(self.ent2id[tgt])
            edge_index[0].append(self.user_mapping[src])
            edge_index[1].append(self.item_mapping[tgt])
            
        self.edge_index = torch.tensor(edge_index).to(self.device)
                
        self.train_edge_index = self.edge_index[:, train_indices].to(self.device)
        self.val_edge_index = self.edge_index[:, val_indices].to(self.device)
        self.test_edge_index = self.edge_index[:, test_indices].to(self.device)
        
        self.num_u = self.interaction_df['source'].nunique()
        self.num_v = self.interaction_df['target'].nunique()
        
        self.train_sparse_edge_index = SparseTensor(row=self.train_edge_index[0], col=self.train_edge_index[1], sparse_sizes=(
            self.num_u + self.num_v, self.num_u + self.num_v)).to(self.device)
        self.val_sparse_edge_index = SparseTensor(row=self.val_edge_index[0], col=self.val_edge_index[1], sparse_sizes=(
            self.num_u + self.num_v, self.num_u + self.num_v)).to(self.device)
        self.test_sparse_edge_index = SparseTensor(row=self.test_edge_index[0], col=self.test_edge_index[1], sparse_sizes=(
            self.num_u + self.num_v, self.num_u + self.num_v)).to(self.device)
        
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
    
    def evaluation(self, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
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
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.model.forward(sparse_edge_index)
        edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
        
        user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

        loss = self.bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                        neg_items_emb_final, neg_items_emb_0, lambda_val).item()

        recall, precision, ndcg = get_metrics_list(self.model, edge_index, exclude_edge_indices, k)

        return loss, recall, precision, ndcg
    
    def fit(self, iterations=10000, batch_size=64, lamb = 1e-6, iters_per_eval=200, items_per_lr_decay=200, K=[1, 5, 10, 15, 20]):
        
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
                val_loss, recalls, precisions, ndcgs = self.evaluation(self.val_edge_index, self.val_sparse_edge_index, [self.train_edge_index], K, lamb)
                print(f"[Iteration {iter}/{iterations}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}")
                print_results(recalls, precisions, ndcgs, K)
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                self.model.train()

            if iter % items_per_lr_decay == 0 and iter != 0:
                self.scheduler.step()
                
        plot_train_val(train_losses, val_losses, iters_per_eval, self.p.name, self.p.n_iter)
        
        # evaluate on test set
        self.model.eval()
        self.test_edge_index = self.test_edge_index.to(self.device)
        self.test_sparse_edge_index = self.test_sparse_edge_index.to(self.device)

        test_loss, test_recall, test_precision, test_ndcg = self.evaluation(
            self.test_edge_index, self.test_sparse_edge_index, [self.train_edge_index, self.val_edge_index], K, lamb)

        print(f"\n\ntest_loss: {round(test_loss, 5)}\n")
        save_results(test_recall, test_precision, test_ndcg, K, self.p.name, self.p.n_iter)
        print_results(test_recall, test_precision, test_ndcg, K)
        
    def predict(self, user_id, num_recs):
        
        user_pos_items = get_user_positive_items(self.edge_index)
        
        user = self.user_mapping[user_id]
        e_u = self.model.users_emb.weight[user]
        scores = self.model.items_emb.weight @ e_u

        values, indices = torch.topk(scores, k=len(user_pos_items[user]) + num_recs)

        items = [index.cpu().item() for index in indices if index in user_pos_items[user]][:num_recs]
        item_ids = [list(self.item_mapping.keys())[list(self.item_mapping.values()).index(item)] for item in items]

        print(f"Here are some items that user {user_id} uses: {item_ids[:num_recs]}")

        items = [index.cpu().item() for index in indices if index not in user_pos_items[user]][:num_recs]
        item_ids = [list(self.item_mapping.keys())[list(self.item_mapping.values()).index(item)] for item in items]

        print(f"Here are some suggested items for user {user_id}: {item_ids[:num_recs]}")