import torch.nn as nn
import torch
import math
import copy
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



class SGCcluster(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, variant):
        super(SGCcluster, self).__init__()
        self.nfeat = nfeat
        self.nlayers = nlayers
        self.nclass = nclass
        self.dropout = dropout
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.polynomial = True
        #self.params2 = list(self.fcs.parameters())


    def forward(self, x, adj, adj_origin):
        device = x.device

        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.fcs[0](x)
        adj = adj.to_dense()
        
        for i in range(self.nlayers):

            if i == 1:
                x = torch.mm(adj, x)
                x = F.dropout(x, self.dropout, training=self.training)               

                temp1 = self.fcs[1](x)
                temp = F.softmax(temp1, dim=1)    # nxc

                L = torch.eye(adj.shape[0]).to(device) - adj
                trace_ = torch.mm(torch.mm(temp.T, L), temp) 

                #if self.polynomial:
                	#L1 = torch.mm(L.T, L)
                	#trace_1 = torch.mm(torch.mm(temp.T, L1), temp)

                

            else:
                x = torch.mm(adj, x)
                x = F.dropout(x, self.dropout, training=self.training)
            '''
            if i == 0:
                adj = self.cluster_spectral(Cluster, x, adj, i)
            '''
            #final = final + x 
        #result = self.fcs[1](x)
        result = temp1

        return torch.log_softmax(result, dim=1), torch.trace(trace_)


if __name__ == '__main__':
    pass






