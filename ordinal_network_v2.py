# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

class ordinal_Network:
    def __init__(self,n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        self.net = nn.Sequential(
            nn.Linear(self.n_input,self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden,self.n_output),
            nn.Tanh()
        )
    
    def fit(self,X,y,sample_weight,learning_rate=0.001):
        sample_weight= torch.tensor(sample_weight)
        #loss_fn = self.my_loss
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        losses = []
        for epoc in range(1000):
            #cost
            hypothesis = self.net(X)
            #cost = ((-y*hypothesis).sum(dim=1) * sample_weight).sum()
            cost = ((y-hypothesis)*(y-hypothesis)*sample_weight).sum()
            
            #loss = torch.max(torch.tensor(0.),hypothesis[:,0:(self.n_output-1)] - hypothesis[:,1:self.n_output]).sum()
            #loss = loss/(len(X)*(self.n_output-1))
            
            loss = torch.max(torch.tensor(0.),hypothesis[:,0:(self.n_output-1)] - hypothesis[:,1:self.n_output]).sum()/(len(X)*(self.n_output-1))

            optimizer.zero_grad()
            cost = cost + loss*(0.5)
            cost.backward()
            
            optimizer.step()

            losses.append(cost.item())
            '''
            if epoc > 20:
                if (losses[epoc-20]-losses[epoc]< 0.001):
                    print("stop learning at %d iteration"%epoc)
                    break
            '''
        
