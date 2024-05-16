import torch
import gc
import torch.nn as nn
from gnn_tr import GNN_prediction, ConvTR
import torch.nn.functional as F
from data import MoleculeDataset, random_sampler
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from sklearn.manifold import TSNE
#from tsnecuda import TSNE # Use this package if the previous one doesn't work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
      
    optimizer_to(optimizer, device)

    return model, optimizer, checkpoint['epoch']

def sample_test(tasks, test_task, data, batch_size, n_support, n_query):
    
    dataset = MoleculeDataset("Data/" + data + "/pre-processed/task_" + str(tasks-test_task), dataset = data)
    support_dataset, query_dataset = random_sampler(dataset, data, tasks-test_task-1, n_support, n_query, train=False)
    support_set = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    query_set = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
    
    return support_set, query_set
    
def roc_accuracy(roc_scores, y_label, y_pred):
    
    roc_auc_list = []
    y_label = torch.cat(y_label, dim = 0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().detach().numpy()
   
    roc_auc_list.append(roc_auc_score(y_label, y_pred))
    roc_auc = sum(roc_auc_list)/len(roc_auc_list)
    roc_scores.append(roc_auc)    
    
    return roc_scores

def parse_pred(logit):
    
    pred = F.sigmoid(logit)
    pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
    pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred) 
    
    return pred

def plot_tsne(nodes, labels, t):
    
    #Plot t-SNE visualizations
    
    labels_tox21 = ['SR-HSE', 'SR-MMP', 'SR-p53']
    labels_sider = ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.P.P.C.']
    labels_list =  labels_sider
    t+=1

    #title = labels_sider[t-1]
        
    emb_tsne = np.asarray(nodes)
    emb_tsne = np.reshape(emb_tsne,(emb_tsne.shape[0],-1))
    y_tsne = np.asarray(labels).flatten()
    
    print(emb_tsne.shape)
      
    c_dict = {'Positive': '#ff7f0e','Negative': '#1f77b4' }
    c = ["#ff8100","#fb9b50","#ffb347","#9fc0de","#0466c8", "#023e7d"]
    z = TSNE(n_components=2, init='random').fit_transform(emb_tsne)
    label_vals = {0: 'Negative', 1: 'Positive'}
    print(y_tsne.size)
    print(z.size)
    tsne_result_df = pd.DataFrame({'tsne_dim_1': z[:,0], 'tsne_dim_2': z[:,1], 'label': y_tsne})
    tsne_result_df['label'] = tsne_result_df['label'].map(label_vals)
    fig, ax = plt.subplots(1)
    
    m_colors = ["blue", "red"]
    for (lab, group), col in zip(tsne_result_df.groupby("label"), m_colors):
       ax.scatter(group.tsne_dim_1, group.tsne_dim_2, edgecolors=col, facecolors="white", alpha = 1, s = 5, linewidth = 0.4, label=lab)
    
    lim = (z.min()-5, z.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal') 
    #fig.set_figwidth(6.5)
    #fig.set_figheight(3.5)
    handles, labels = ax.get_legend_handles_labels()
    if t == 1:
        ax.legend(handles[:2], labels[:2], bbox_to_anchor=(1, 0.02), loc='lower right')
    ax.set_title(labels_list[t-1])
    ax.set(xlabel="Dimension 1")
    ax.set(ylabel="Dimension 2")
    ax.tick_params(bottom=True) 
    ax.tick_params(left=True)
    plt.grid(False)
    plt.savefig('plots/'+labels_list[t-1], dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return t

class GNNCvTR_eval(nn.Module):
    def __init__(self, dataset, gnn, support_set, pretrained, baseline):
        super(GNNCvTR_eval,self).__init__()
                
        if dataset == "tox21":
            self.tasks = 12
            self.train_tasks = 9 
            self.test_tasks = 3 

        elif dataset == "sider":
            self.tasks = 27
            self.train_tasks = 21 
            self.test_tasks = 6 
            
        self.data = dataset
        self.baseline = baseline
        self.graph_layers = 5
        self.n_support = support_set
        self.learning_rate = 0.001
        self.n_query = 128
        self.emb_size = 300
        self.batch_size = 10
        self.lr_update = 0.4
        self.k_train = 5
        self.k_test = 10
        self.device = 0
        self.gnn = GNN_prediction(self.graph_layers, self.emb_size, jk = "last", dropout_prob = 0.5, pooling = "mean", gnn_type = gnn)
        self.transformer = ConvTR() 
        self.gnn.from_pretrained(pretrained)
        self.pos_weight = torch.FloatTensor([25]).to(self.device) #Tox21: 25; SIDER: 1
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.loss_transformer = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.meta_opt = torch.optim.Adam(self.transformer.parameters(), lr=1e-5)
        
        graph_params = []
        graph_params.append({"params": self.gnn.gnn.parameters()})
        graph_params.append({"params": self.gnn.graph_pred_linear.parameters(), "lr":self.learning_rate})
        
        self.optimizer = optim.Adam(graph_params, lr=self.learning_rate, weight_decay=0) 
        self.gnn.to(torch.device("cuda:0"))
        self.transformer.to(torch.device("cuda:0"))
        
        if (self.baseline == 0):
            self.ckp_path_gnn = "checkpoints/checkpoints-GT/FS-GNNCvTR_GNN_sider_5.pt"
            self.ckp_path_transformer = "checkpoints/checkpoints-GT/FS-GNNCvTR_Transformer_sider_5.pt"
        elif (self.baseline == 1):
            self.ckp_path_gnn = "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_tox21_05.pt"
        
        self.gnn, self.optimizer, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.optimizer)
        self.transformer, self.meta_optimizer, start_epoch = load_ckp(self.ckp_path_transformer, self.transformer, self.meta_optimizer)
        
 
    def update_graph_params(self, loss, lr_update):
        grads = torch.autograd.grad(loss, self.gnn.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.gnn.parameters()) - parameters_to_vector(grads) * lr_update

    def meta_evaluate(self):
        
        roc_scores = []

        t=0
        graph_params = parameters_to_vector(self.gnn.parameters())
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        
        for test_task in range(self.test_tasks):
            
            support_set, query_set = sample_test(self.tasks, test_task, self.data, self.batch_size, self.n_support, self.n_query)
            self.gnn.eval()
            if self.baseline == 0:
                self.transformer.eval()
                
            for k in range(0, self.k_test):
                
                graph_loss = torch.tensor([0.0]).to(device)
                
                if self.baseline == 0:
                    loss_logits = torch.tensor([0.0]).to(device)
                
                for batch_idx, batch in enumerate(tqdm(support_set, desc="Iteration")):
                    
                    batch = batch.to(device)
                    graph_pred, emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y = batch.y.view(graph_pred.shape)
                    loss_graph = self.loss(graph_pred.double(), y.to(torch.float64))
                    graph_loss += torch.sum(loss_graph)/graph_pred.size(dim=0)
                    
                    if self.baseline == 0:
                        
                        with torch.no_grad():
                            val_logit, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                        
                        loss_tr = self.loss_transformer(F.sigmoid(val_logit).double(), y.to(torch.float64))
                        loss_logits += torch.sum(loss_tr)/val_logit.size(dim=0)
                          
                    del graph_pred, emb
                    
                updated_grad, updated_params = self.update_graph_params(graph_loss, lr_update = self.lr_update)
                vector_to_parameters(updated_params, self.gnn.parameters())
            
            torch.cuda.empty_cache()
            
            nodes=[]
            labels=[]
            y_label = []
            y_pred = []
           
            for batch_idx, batch in enumerate(tqdm(query_set, desc="Iteration")):
                batch = batch.to(device)
                
                with torch.no_grad(): 
                    logit, emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
                y_label.append(batch.y.view(logit.shape))
            
                if self.baseline == 0:
                    with torch.no_grad(): 
                        logit, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                
                #print(F.sigmoid(logit))
          
                pred = parse_pred(logit)
                
                emb_tsne = emb.cpu().detach().numpy() 
                y_tsne = batch.y.view(pred.shape).cpu().detach().numpy()
               
                for i in emb_tsne:
                    nodes.append(i)
                for j in y_tsne:
                    labels.append(j)
                
                y_pred.append(pred)   
                
            t = plot_tsne(nodes, labels, t)
             
            roc_scores  = roc_accuracy(roc_scores, y_label, y_pred)
            
            vector_to_parameters(graph_params, self.gnn.parameters())
                
        return roc_scores, self.gnn.state_dict(), self.transformer.state_dict(), self.optimizer.state_dict(), self.meta_optimizer.state_dict()
        #return [statistics.mean(roc_scores)], self.gnn.state_dict(), self.transformer.state_dict(), self.optimizer.state_dict(), self.meta_optimizer.state_dict()    
