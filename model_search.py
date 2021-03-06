import pandas as pd 
import numpy as np 
import os 
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import  roc_auc_score, accuracy_score ,f1_score
from sklearn.pipeline import Pipeline 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk 
st = stopwords.words('english')
stemmer = PorterStemmer()
import sys 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from nltk.tokenize import sent_tokenize
import pickle 
from utils import *





data = pd.read_csv('pdfalls.csv')
data['labels'] = data['fall_class'] =='CoM' 
class_com= np.sum(data['labels'])
class_other = np.sum(np.logical_not(data['labels'] )) 
print(f'We have {class_com} class CoM')
print(f'We have {class_other} class other')
vectorizer = CountVectorizer(ngram_range=(1,2)) 
my_models = [DecisionTreeClassifier] 
subj =  {'subj':1,'obj':2}
senti = pickle.load(open('senti_analyzer.pkl','rb') )
feat_extractor = lambda x: np.array([subj[senti.classify(e)] for e in x ]).reshape(-1,1)
model_params = {} 
model_params['tree']=   { 'tree__criterion':['gini','entropy'],'tree__max_depth':np.linspace(10,30,5,dtype=np.int)}
model_params['forest'] ={'forest__n_estimators':np.linspace(5,20,5,dtype=np.int),'forest__max_depth':range(4,8)}
model_params['naive'] = {'naive__fit_prior':[True,False],'naive__alpha':np.linspace(0,1,5)}
model_params['svm'] = {'svm__C':np.linspace(0.2 , 200,10)}
model_params['logi'] = {'logi__C':np.linspace(0.2,200,10)}
model_params['quadri'] = {'quadri__reg_param':np.linspace(0,1,20)}
models = {'tree':DecisionTreeClassifier(),'forest':RandomForestClassifier(),'naive':MultinomialNB(),'svm':svm.SVC(),'logi':LogisticRegression(),'quadri':QuadraticDiscriminantAnalysis()}
for e in model_params.keys():
    print(f"We are going to try out {e} ----------")
    model = models[e]
    params:dict = model_params[e]  
    params.update({'vect__counts__max_features':np.linspace(100,200,10,dtype=np.int), 'vect__tfidf__max_features':np.linspace(100,200,10,dtype=np.int)})
    ct = ColumnTransformer(
        [('counts',CountVectorizer(),'fall_description'),
    ('tfidf',TfidfVectorizer(),'simplified_location'),
    ('encoding', FunctionTransformer(func_transformer),'simplified_location'),
     ('polarity',FunctionTransformer(extract_polarity) ,'fall_description' ),
       ('subjectivity',FunctionTransformer(feat_extractor),'fall_description')
    ]  ) 
    other=  ct.fit_transform(data) 
    pipe = Pipeline(steps=[('vect',ct),(e,models[e] ) ] ) 
    search = GridSearchCV( pipe,params, n_jobs= 12, scoring='f1_micro',verbose=1,cv=3) 
    search.fit(data,data['labels']) 
    print(search.best_estimator_)
    print(search.best_score_)
    print(search.best_params_)

