import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from itertools import combinations
from utils import *  
from sklearn.model_selection import RepeatedKFold 
from sklearn.metrics import accuracy_score, f1_score
from typing import List 


# Accuracy, micro-averaged F1 score, and macro-averaged F1 score

#k{'forest__max_depth': 7, 'forest__n_estimators': 5, 'vect__counts__max_features': 122, 'vect__tfidf__max_features': 177}
if __name__ == "__main__": 
    data = pd.read_csv('pdfalls.csv') 
    featList = [('counts',CountVectorizer(max_features=122),'fall_description'),
    ('tfidf',TfidfVectorizer(max_features=177),'simplified_location'),
    ('encoding', FunctionTransformer(func_transformer),'simplified_location'),
     ('polarity',FunctionTransformer(extract_polarity) ,'fall_description' ),
       ('subjectivity',FunctionTransformer(get_subj_func()),'fall_description')
    ]  
    model = RandomForestClassifier(n_estimators=5,max_depth=7,n_jobs=5)
    selector = RepeatedKFold(n_splits=3,n_repeats=3)
    data['labels'] = data['fall_class'] =='CoM'
    for k in range(2,5): 
        print(f"===== for k = {k} =====") 
        findings = list() 
        for combi in combinations(featList,k):    
            ct = ColumnTransformer(combi,n_jobs=2) 
            pipe = Pipeline(steps=[('vect',ct),('forest',model ) ]) 
            out_labels  = list() 
            truth_labels = list()   
            for train, test in selector.split(data,data['labels']): 
                pipe.fit(data.iloc[train,],data['labels'][train])
                out_labels.append( pipe.predict(data.iloc[test])) 
                truth_labels.append(data['labels'][test])
            metrics = gen_metrics(out_labels,truth_labels)
            findings.append((metrics[0], metrics[1], metrics[2],combi))
        findings = sorted(findings,key =lambda x: x[1]) #sorted in increasing order 
        print(f" Best performance fo k={k} is {findings[-1]} ") 