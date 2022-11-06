# -*- coding: utf-8 -*-
import numpy as np
import torch
from ordinal_network_v2 import ordinal_Network

import time

class Ordinal_boosting:
    
    def __init__(self, n_estimators=10, n_hidden=6, learning_rate=0.001):
        self.n_estimators = n_estimators
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
    
    def encoding_y(self,y):
        self.set_class = set(y)
        self.n_class = len(set(y))
        en = np.repeat(1,self.n_class**2)
        en = en.reshape(self.n_class,-1)
        for i in range(self.n_class):
            if i > 0:
                en[i,0:i] = -1
        d = {}
        for i,j in enumerate(tuple(set(y))):
            d[j] = en[i,:]
        self.enco_y = d
        return self
    
    def fit(self,X,y):
        N = len(y)
        self.n_input = X.shape[1]
        self.y_min = min(y)
        self.estimator_list, self.g_vector_list, self.alpha_list, self.sample_weight_list = [],[],[],[]

        #y encoding
        self.encoding_y(y)
        enco_tr_Y = list(map(lambda x: self.enco_y[x], y))
        enco_tr_Y = np.array(enco_tr_Y)
        Y_tr = torch.tensor(enco_tr_Y, dtype=torch.float32)
        
        #Initialize the sample weights
        sample_weight = np.ones((N,self.n_class)) / (N*self.n_class)
        self.sample_weight_list.append(sample_weight.copy())
        
        #For m = 1 to M
        for m in range(self.n_estimators):   

            #print("_______________________________________________________________________________")
            #print("number of ANN: %d"%m)
            #start = time.time()
            
            #Fit a classifier
            #Neural Network
            network = ordinal_Network(self.n_input,self.n_hidden,self.n_class)
            network.fit(X,Y_tr,sample_weight, self.learning_rate)
            
            '''
            if m == 0:
                network.fit(X,Y_tr,sample_weight,False)
            else:
                network.fit(X,Y_tr,sample_weight,self.estimator_list[m-1].net.state_dict())
            '''
            
            # iteration time
            #print("time :", time.time() - start)
            
            y_predict = network.net(X)
            
            # nn output to encoding 
            pre_y = np.zeros(len(y_predict))
            for i in range(len(y_predict)):
                score = {}
                for j in self.enco_y.keys():
                    score[j] = np.sum(y_predict.detach().numpy()[i]*self.enco_y[j])
                pre_y[i] = max(score, key=score.get)
            
            g_vector = list(map(lambda x: self.enco_y[x], pre_y))
            g_vector = np.array(g_vector)
            
            # Error
            err = ((enco_tr_Y != g_vector)*sample_weight).sum()/(sample_weight.sum())
            if err < 0.001:
                err = 0.001
            
            alpha = np.log((1.- err)/err)/2
            
            if alpha < 0:
                alpha = 0.

            sample_weight *= ((enco_tr_Y == g_vector)*np.exp(-alpha)+(enco_tr_Y != g_vector)*np.exp(alpha))
            sample_weight = sample_weight/np.sum(sample_weight)
            
            self.estimator_list.append(network)
            self.g_vector_list.append(g_vector)
            self.alpha_list.append(alpha)
            self.sample_weight_list.append(sample_weight.copy())

        #Convert to np array for convenience   
        #estimator_list = np.asarray(estimator_list)
        self.g_vector_list = np.asarray(self.g_vector_list)
        self.alpha_list = np.asarray(self.alpha_list)
        self.sample_weight_list = np.asarray(self.sample_weight_list)
        
        #Predictions
        f_vector = np.zeros(shape = self.g_vector_list[0].shape)
        for m in range(self.n_estimators):
            f_vector += self.g_vector_list[m]*self.alpha_list[m]
        
        p = np.exp(2*f_vector)/(1+np.exp(2*f_vector))
        s = np.zeros(p.shape)
        s[:,1:] = p[:,0:(p.shape[1]-1)]
        preds = np.argmax(p-s,axis=1) + self.y_min

        self.train_acc = (preds == y).sum()/N
        self.train_mae = np.sum(np.abs(preds-y))/N
        #print('Ordinal_boosting_train_Accuracy = ', self.train_acc)
        #print('Ordinal_boosting_train_MAE = ', self.train_mae)
        
        return self
        
    def predict(self,X_test,n_estimators_test=10):
        #with torch.no_grad()
        f_vector_test = np.zeros(shape = (len(X_test),self.n_class))
        for m in range(n_estimators_test):
            y_predict = self.estimator_list[m].net(X_test)
            
            score = np.zeros(shape = (len(X_test),self.n_class))
            with torch.no_grad():
                for i,j in enumerate(self.enco_y.keys()):
                    score[:,i] = torch.matmul(y_predict, torch.tensor(self.enco_y[j], dtype=torch.float32))
                pre_y = np.argmax(score,axis=1) + self.y_min

            g_vector = list(map(lambda x: self.enco_y[x], pre_y))
            g_vector = np.array(g_vector)
            f_vector_test += g_vector*self.alpha_list[m]
        
        #preds = (f_vector_test < 0).sum(axis=1) + self.y_min
        #preds = np.array(preds)

        p = np.exp(2*f_vector_test)/(1+np.exp(2*f_vector_test))
        s = np.zeros(p.shape)
        s[:,1:] = p[:,0:(p.shape[1]-1)]
        preds = np.argmax(p-s,axis=1) + self.y_min
        
        return(preds)
        
