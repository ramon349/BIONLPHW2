import pandas as pd 
import numpy as np 
import os 
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import  roc_auc_score, accuracy_score ,f1_score
from sklearn.pipeline import Pipeline 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
st = stopwords.words('english')
stemmer = PorterStemmer()
import sys 

def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    #stemming and lowercasing (no stopword removal
    words = [w for w in raw_text.lower().split()]
    return (" ".join(words))


data = pd.read_csv('pdfalls.csv')
data['labels'] = data['fall_class'] =='CoM' 
class_com= np.sum(data['labels'])
class_other = np.sum(np.logical_not(data['labels'] )) 
print(f'We have {class_com} class CoM')
print(f'We have {class_other} class other')



vectorizer = CountVectorizer(ngram_range=(1,2)) 
my_models = [DecisionTreeClassifier]
model_params = {} 
model_params['tree']=   { 'tree__criterion':['gini','entropy'],'tree__max_depth':np.linspace(10,100,20,dtype=np.int)}
model_params['forest'] ={'forest__n_estimators':np.linspace(10,50,50,dtype=np.int),'forest__max_depth':range(4,10)}
model_params['naive'] = {'naive__fit_prior':[True,False],'naive__alpha':np.linspace(0,1,10)}
models = {'tree':DecisionTreeClassifier(),'forest':RandomForestClassifier(),'naive':MultinomialNB() }


print(model_params.keys())

for e in model_params.keys():
    print(f"We are going to try out {e} ----------")
    model = models[e]
    params = model_params[e] 
    pipe = Pipeline(steps=[('vect',vectorizer),(e,model) ] ) 
    param_grid = {'vect__max_features':np.linspace(1000,2000,200,dtype=np.int)}
    param_grid.update(params)
    search = GridSearchCV( pipe,param_grid, scoring='f1',n_jobs=-1,verbose=0) 
    search.fit(data['fall_description'],data['labels']) 
    print(f"Best parameters were {search.best_estimator_}") 

