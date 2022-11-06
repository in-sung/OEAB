# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_dataset(dataname: str, test_size: float = 0.25):
    try:
        assert dataname in ['toy','wine','car','tae','boston','balance','machine','stock']
    except AssertionError as e:
        raise
        
    if dataname == 'toy':
        np.random.seed(seed=123)
        rv = np.random.uniform(0,1,1000)
        
        np.random.seed(seed=123)
        err = np.random.normal(0,0.125,500)
        x1 = rv[0:500]
        x2 = rv[500:1000]
        
        def cal_y(x1,x2,err):
            if 10*(x1-0.5)*(x2-0.5)+err < -1:
                y = 1
            elif 10*(x1-0.5)*(x2-0.5)+err < -0.1:
                y = 2
            elif 10*(x1-0.5)*(x2-0.5)+err < 0.25:
                y = 3
            elif 10*(x1-0.5)*(x2-0.5)+err < 1:
                y = 4
            else:
                y = 5
            return(y)
        
        y = list(map(cal_y,x1,x2,err))
        y = np.array(y)
        X = np.vstack((x1,x2)).T
               
    elif dataname == 'wine':
        df = pd.read_csv('./ordinal_data/winequality-red.csv')      
        X = df.drop(['quality'],axis=1)
        y = np.array(df['quality'])
        
    elif dataname == 'car':
        df = pd.read_csv('./ordinal_data/car_data.txt',sep=',',names=['buying','maint','doors','persons','lug_boot','safety','class'] ,header=None)
        enc = OneHotEncoder()
        enc.fit(df.drop(['class'],axis=1))
        X = enc.transform(df.drop(['class'],axis=1)).toarray()
        
        y = np.zeros(len(df))

        for i in range(len(y)):
            if df.loc[i,'class'] == 'unacc':
                y[i] = 1
            elif df.loc[i,'class'] == 'acc':
                y[i] = 2
            elif df.loc[i,'class'] == 'good':
                y[i] = 3
            elif df.loc[i,'class'] == 'vgood':
                y[i] = 4

    elif dataname == 'tae':
        df = pd.read_csv('./ordinal_data/tae_data.txt',sep=',',names=['English','instructor','Course','Summer','size','attribute'],header=None )
        enc = OneHotEncoder()
        enc.fit(df.drop(['size','attribute'],axis=1))
        
        X = pd.DataFrame(enc.transform(df.drop(['size','attribute'],axis=1)).toarray())
        X = pd.concat([X,df['size']],axis=1)
        y = np.array(df['attribute'])
    
    elif dataname == 'boston':
        df = pd.read_csv('./ordinal_data/bostonhousing5.txt',sep=',')
        X = df.drop(['response'],axis=1)
        y = np.array(df['response'])
        
    elif dataname == 'balance':
        df = pd.read_csv('./ordinal_data/balance-scale.txt',sep=',',names=['Class_Name','Left-Weight','Left-Distance','Right-Weight','Right-Distance'] ,header=None )
        X = df.drop(['Class_Name'],axis=1)
        
        y = np.zeros(len(df))
        for i in range(len(y)):
            if df.loc[i,'Class_Name'] == 'L':
                y[i] = 1
            elif df.loc[i,'Class_Name'] == 'B':
                y[i] = 2
            elif df.loc[i,'Class_Name'] == 'R':
                y[i] = 3
        
    elif dataname == 'machine':
        df = pd.read_csv('./ordinal_data/machine.txt',sep=',')
        X = df.drop(['response'],axis=1)
        y = np.array(df['response'])
        
    elif dataname == 'stock':
        df = pd.read_csv('./ordinal_data/stock5.txt',sep=',')
        X = df.drop(['response'],axis=1)
        y= np.array(df['response'])
        
        
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_tr_scaled = scaler.transform(x_train)
    X_te_scaled = scaler.transform(x_test)
        
    return(X_tr_scaled, X_te_scaled, y_train, y_test)
        
        
        
        
        
    