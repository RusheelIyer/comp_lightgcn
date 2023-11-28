from helper import *
from data_loader import TrainDataset, TestDataset, TestBCEDataset
from model.models import CompGCN_TransE, CompGCN_DistMult, CompGCN_ConvE, CompGCN_BCE

from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split

class CompGCNEngine(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format. 

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)
        
        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        user_set, item_set = OrderedSet(), OrderedSet()
        
        df = pd.read_csv('data/graph.csv')
        
        for _, row in df.iterrows():
            sub, rel, obj = row['source'], row['relation'], row['target']
            if rel != 'uses':
                ent_set.add(sub)
                rel_set.add(rel)
            else:
                user_set.add(sub)
                item_set.add(obj)
                
            ent_set.add(obj)
        
        knowledge_df = df[df['relation'] != 'uses']
        
        k_train, k_temp = train_test_split(knowledge_df, test_size=0.2)
        k_valid, k_test = train_test_split(k_temp, test_size=0.5)
        
        interaction_df = df[df['relation'] == 'uses']
        self.i_train, i_temp = train_test_split(interaction_df, test_size=0.2)
        self.i_valid, self.i_test = train_test_split(i_temp, test_size=0.5)

        self.user2id = {user: idx for idx, user in enumerate(user_set)}
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2user = {idx: user for user, idx in self.user2id.items()}
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_users		= len(user_set)
        self.p.num_items		= len(item_set)
        self.p.num_ent		    = len(self.ent2id)
        self.p.num_rel		    = len(self.rel2id) // 2
        self.p.embed_dim	    = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)
        
        # Normal Splits
        for _, row in k_train.iterrows():
            sub, rel, obj = row['source'], row['relation'], row['target']
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
            self.data['train'].append((sub, rel, obj))
            
            sr2o[(sub, rel)].add(obj)
            sr2o[(obj, rel+self.p.num_rel)].add(sub)
            
        for _, row in k_test.iterrows():
            sub, rel, obj = row['source'], row['relation'], row['target']
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
            self.data['test'].append((sub, rel, obj))
            
        for _, row in k_valid.iterrows():
            sub, rel, obj = row['source'], row['relation'], row['target']
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
            self.data['valid'].append((sub, rel, obj))
            
        # BCE Splits
        for _, row in knowledge_df.iterrows():
            sub, rel, obj = row['source'], row['relation'], row['target']
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
            self.data['train_bce'].append((sub, rel, obj))
        
        for _, row in self.i_valid.iterrows():
            sub, obj = row['source'], row['target']
            sub, obj = self.user2id[sub], self.ent2id[obj]
            self.data['valid_bce'].append((sub, -1, obj))
            
        for _, row in self.i_test.iterrows():
            sub, obj = row['source'], row['target']
            sub, obj = self.user2id[sub], self.ent2id[obj]
            self.data['test_bce'].append((sub, -1, obj))

        self.data = dict(self.data)
        self.triples  = ddict(list)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
            self.triples['train_bce'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})
                
        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return  DataLoader(
                    dataset_class(self.triples[split], self.p),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.p.num_workers),
                    collate_fn      = dataset_class.collate_fn
                ) if split != 'train_bce' else DataLoader(
                    dataset_class(self.triples[split], self.p),
                    batch_size      = 1024,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.p.num_workers),
                    collate_fn      = dataset_class.collate_fn
                )

        self.data_iter = {
            'train':    			get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
            'valid_head':   		get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
            'valid_tail':   		get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
            'test_head':   			get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
            'test_tail':   			get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
            
            'train_bce':    		get_data_loader(TrainDataset, 'train_bce',  self.p.batch_size),
            'valid_bce':   		    DataLoader(
                                            TestBCEDataset(self.data['valid_bce'], self.p),
                                            batch_size = self.p.batch_size,
                                            shuffle = True,
                                            num_workers = max(0, self.p.num_workers)
                                    ),
            'test_bce':   			DataLoader(
                                            TestBCEDataset(self.data['test_bce'], self.p),
                                            batch_size = self.p.batch_size,
                                            shuffle = True,
                                            num_workers = max(0, self.p.num_workers)
                                    )
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------
        
        Returns
        -------
        Constructs the adjacency matrix for GCN
        
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index	= torch.LongTensor(edge_index).to(self.device).t()
        edge_type	= torch.LongTensor(edge_type). to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model
        
        Returns
        -------
        Creates computational graph and optimizer
        
        """
        self.p      = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model        = self.add_model(self.p.model, self.p.score_func)
        self.optimizer    = self.add_optimizer(self.model.parameters())


    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created
        
        Returns
        -------
        Creates the computational graph for model and initializes it
        
        """
        model_name = '{}_{}'.format(model, score_func)

        if   model_name.lower()	== 'compgcn_transe': 	model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower()	== 'compgcn_distmult': 	model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower()	== 'compgcn_conve': 	model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower()	== 'compgcn_bce': 	    model = CompGCN_BCE(self.edge_index, self.edge_type, params=self.p)
        else: raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model
        
        Returns
        -------
        Returns an optimizer for learning the parameters of the model
        
        """
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split

        
        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split == 'train':
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved
        
        Returns
        -------
        """
        state = {
            'state_dict':   self.model.state_dict(),
            'best_val':     self.best_val,
            'best_epoch':   self.best_epoch,
            'optimizer':    self.optimizer.state_dict(),
            'args':         vars(self.p),
            'item_embs':    self.model.item_embed,
            'user_embs':    self.model.user_embeddings
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model
        
        Returns
        -------
        """
        state			    = torch.load(load_path, map_location=self.device)
        state_dict		    = state['state_dict']
        self.best_val		= state['best_val']
        # self.best_val_mrr	= self.best_val['mrr']
        self.item_embed     = state['item_embs']
        self.user_embed     = state['user_embs']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count
        
        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        if 'bce' not in split:
            left_results  = self.predict(split=split, mode='tail_batch')
            right_results = self.predict(split=split, mode='head_batch')
            results       = get_combined_results(left_results, right_results)
            self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
            return results
        else:
            self.model.eval()
            
            with torch.no_grad():
                results = {}
                random.seed(42)
                
                for _, batch in enumerate(iter(self.data_iter[split])):
                    sub = batch[:,0]
                    obj = batch[:,2]
                    
                    def sample_neg_items_for_u(u, num):
                        pos_items = set(self.i_valid[self.i_valid['source'] == u]['target'])
                        all_items = set(self.i_valid['target'].unique())
                        neg_items = list(all_items - pos_items)
                        
                        if len(neg_items) == 0:
                            return []
                        
                        neg_items = [self.ent2id[item] for item in neg_items]
                        
                        return random.sample(neg_items, num) if len(neg_items) >= num else random.choices(neg_items, num)
                    
                    neg_items = [sample_neg_items_for_u(self.id2ent[user.int().item()], 1)[0] for user in sub]

                    users = self.model.user_embeddings[sub]
                    pos_items = self.model.item_embed[obj]
                    neg_items = self.model.item_embed[neg_items]
                    
                    loss = self.model.bce_loss(users, pos_items, neg_items)
                    return sum(loss)
                    

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'
        
        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label	= self.read_batch(batch, split)
                pred			= self.model.forward(sub, rel)
                b_range			= torch.arange(pred.size()[0], device=self.device)
                target_pred		= pred[b_range, obj]
                pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] 	= target_pred
                ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

                ranks 			= ranks.float()
                results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
                results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
                results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results

    def sample_batch(self, batch_size):
        
        users = list(self.i_train['source'].unique())
        n_users = len(users)
        
        random.seed(42)
        if batch_size <= n_users:
            users = random.sample(users, batch_size)
        else:
            users = [random.choice(users) for _ in range(batch_size)]
            
        def sample_pos_items_for_u(u, num):
            pos_items = list(self.i_train[self.i_train['source'] == u]['target'])
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_item_id = self.ent2id[pos_items[pos_id]]

                if pos_item_id not in pos_batch:
                    pos_batch.append(pos_item_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            pos_items = set(self.i_train[self.i_train['source'] == u]['target'])
            all_items = set(self.i_train['target'].unique())
            neg_items = list(all_items - pos_items)
            
            if len(neg_items) == 0:
                return []
            
            neg_items = [self.ent2id[item] for item in neg_items]
                    
            return random.sample(neg_items, num) if len(neg_items) >= num else random.choices(neg_items, num)
        
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            
        users = [self.user2id[user] for user in users]
        
        return (users, pos_items, neg_items)
        

    def run_epoch(self, epoch, val_mrr = 0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count
        
        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train']) if self.p.score_func.lower() != 'bce' else iter(self.data_iter['train_bce'])
        
        if self.p.score_func.lower() == 'bce':
            for _ in range(self.p.bce_iter):
                self.optimizer.zero_grad()
                batch = self.sample_batch(self.p.batch_size)
                for step, k_batch in enumerate(train_iter):
                    sub, rel, _, _ = self.read_batch(k_batch, 'train')
                    mf_loss, emb_loss, reg_loss	= self.model.forward(batch, sub, rel)
                    
                    loss = mf_loss + emb_loss + reg_loss
                
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

                    if step % 100 == 0:
                        self.logger.info('[E:{}| {}]: Train Loss:{:.5}, \t{}'.format(epoch, step, np.mean(losses), self.p.name))
                
        else:
            for step, batch in enumerate(train_iter):
                self.optimizer.zero_grad()
                sub, rel, obj, label = self.read_batch(batch, 'train')
                print(sub)
                print(rel)

                pred	= self.model.forward(sub, rel)
                loss	= self.model.loss(pred, label)

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                if step % 100 == 0:
                    self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss


    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------
        
        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        self.best_val_loss = None
        save_path = os.path.join(self.p.checkpoint_dir, self.p.name)
        if not os.path.exists(self.p.checkpoint_dir):
            os.mkdir(self.p.checkpoint_dir)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss  = self.run_epoch(epoch, val_mrr)
            
            if self.p.score_func.lower() != 'bce':
                
                val_results = self.evaluate('valid', epoch)

                if val_results['mrr'] > self.best_val_mrr:
                    self.best_val	   = val_results
                    self.best_val_mrr  = val_results['mrr']
                    self.best_epoch	   = epoch
                    self.item_embed = self.model.item_embed
                    self.save_model(save_path)
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt % 10 == 0 and self.p.gamma > 5:
                        self.p.gamma -= 5 
                        self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > 25: 
                        self.logger.info("Early Stopping!!")
                        break

                self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))
                
            else:
                val_loss = self.evaluate('valid_bce', epoch)
                
                if self.best_val_loss is None or val_loss <= self.best_val_loss:
                    self.best_val	   = {'val_loss': val_loss}
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt % 10 == 0 and self.p.gamma > 5:
                        self.p.gamma -= 5 
                        self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > 25: 
                        self.logger.info("Early Stopping!!")
                        break
                    
                self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid Loss: {:.5}\n\n'.format(epoch, train_loss, val_loss))

        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        
        if self.p.max_epochs > 0:
            test_results = self.evaluate('test', epoch)