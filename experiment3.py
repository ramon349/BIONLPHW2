import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from itertools import combinations
from utils import *  
from sklearn.model_selection import LeavePOut, train_test_split
from sklearn.metrics import accuracy_score, f1_score 
from typing import List 
import matplotlib.pyplot as plt 
#k{'forest__max_depth': 7, 'forest__n_estimators': 5, 'vect__counts__max_features': 122, 'vect__tfidf__max_features': 177}
if __name__ == "__main__": 
    data = pd.read_csv('pdfalls.csv') 
    featList = [
    ('counts',CountVectorizer(max_features=122),'simplified_location'),
     ('polarity',FunctionTransformer(extract_polarity) ,'fall_description' ),
       ('subjectivity',FunctionTransformer(get_subj_func()),'fall_description')
    ]  
    model = RandomForestClassifier(n_estimators=5,max_depth=7,n_jobs=5)
    data['labels'] = data['fall_class'] =='CoM'
    data_size = data.shape[0]
    acc = list() 
    f1_micro= list() 
    f1_macro = list() 
    for k in np.arange(.2,.9,.05): 
        amount = int( np.round( data_size*k)  )
        test_size =  int( np.round( data_size*k) )
        train_size =  data_size - test_size
        print(f"===== for k = {amount} =====") 
        ct = ColumnTransformer(featList,n_jobs=5) 
        pipe = Pipeline(steps=[('vect',ct),('forest',model ) ]) 
        out_labels  = list() 
        truth_labels = list()   
        X_train, X_test, y_train, y_test = train_test_split(data, data['labels'], test_size=k)
        pipe.fit( X_train,y_train)
        out_labels.append( pipe.predict(X_test) )
        truth_labels.append(y_test)
        metrics = gen_metrics(out_labels,truth_labels) 
        print( [f"{name}:{val}" for name,val in zip(['acc','f1_micro','f1_macro' ],metrics ) ])
        acc.append(metrics[0])
        f1_micro.append(metrics[1])
        f1_macro.append(metrics[2])
    x_val = np.arange(.2,.9,.05) 
    plt.plot(x_val,acc,'ro')  
    plt.plot(x_val,f1_micro,'gx')
    plt.plot(x_val,f1_macro,'bv')
    plt.xlabel('Test Set size (percentage)')
    plt.ylabel('Model Performance ')
    plt.legend(['accuracy','f1_micro','f1_macro'])
    plt.title("Model Peformance based while varying train size")
    plt.show()