from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=72, help='Random seed.')  # 42
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=2, help='device id')
parser.add_argument('--mu', type=float, default=0.003, help='mu for cora, 3e-5 for citeseer, 6e-5 for pubmed')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels,idx_train,idx_val,idx_test,  adj_origin, graph = load_citation(args.data)
#adj, features, labels,idx_train,idx_val,idx_test, adj_origin, graph = load_highfrequency(args.data)

print(adj.shape)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
#adj_origin = adj_origin.to(device)

checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)




model = SGCcluster(nfeat=features.shape[1],
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                variant=args.variant).to(device)


optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.wd2)

 

def train():
    model.train()
    optimizer.zero_grad()
    
    output, trace = model(features, adj, adj_origin)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train1 = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    
    loss_train = loss_train1 + args.mu * trace

    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        
        output, trace = model(features, adj, adj_origin)
        loss_val1 = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        
        loss_val = loss_val1 + args.mu * trace
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        
        output, trace = model(features, adj, adj_origin)
        loss_test1 = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        
        loss_test = loss_test1 + args.mu * trace
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()
    
t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra,acc_tra = train()
    loss_val,acc_val = validate()
    if(epoch+1)%1 == 0: 
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            'acc:{:.2f}'.format(acc_tra*100),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100))
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
'''
f = open("result/{}_regtrace&mu{}.txt".format(args.data, args.mu),'a')
f.write("lr:{}, wd:{}, layer:{}, acc_test: {}".format(args.lr, args.wd2, args.layer, acc*100))
f.write("\n")
f.close()
'''

