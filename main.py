# -*- coding: utf-8 -*-
import numpy as np
import torch

from ordinal_boosting_v3 import Ordinal_boosting
from dataset import load_dataset

from sklearn.metrics import f1_score
import argparse


def main():
    torch.manual_seed(args.seed)
    
    # load data
    x_train, x_test, y_train, y_test = load_dataset(dataname=args.dataname, test_size= args.test_size)
    X_tr = torch.tensor(x_train, dtype=torch.float32)
    X_te = torch.tensor(x_test, dtype=torch.float32)
    
    model = Ordinal_boosting(n_estimators=args.n_estimators,n_hidden=args.hidden_node,
                                learning_rate=args.learning_rate)
    model.fit(X_tr,y_train)
    y_pred = model.predict(X_te, n_estimators_test=args.n_estimators)
    
    print("test accuracy: ",(y_pred == y_test).sum() / len(y_test))
    print("test MAE: ",np.sum(np.abs(y_pred-y_test)) / len(y_test))
    print("test f1-score: ",f1_score(y_test, y_pred, average='macro'))


if __name__ == '__main__':
    #set arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=2227,
                        help="Specify random state")
    
    # data
    parser.add_argument('--dataname',type=str, default='toy',
                        choices=['toy','wine','car','tae','boston','balance','machine','stock'])
    
    parser.add_argument('--test_size',type=float, default=0.25,
                        help="test size")    

    # model options
    parser.add_argument('--n_estimators', type=int, default=10,
                        help="# of estimators")
    
    parser.add_argument('--hidden_node', type=int, default=16,
                        help="# of hidden node of NN")

    # train options
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help="learning rate of Adam Optimizer")   
    
    args = parser.parse_args()
    
    main()