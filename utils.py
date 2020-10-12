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
from typing import List
def gen_metrics(runs:List,truth:List):
    acc = list() 
    f1_micro = list() 
    f1_macro = list() 
    for pred,truth in zip(runs,truth): 
        acc.append( accuracy_score(pred,truth) ) 
        f1_micro.append(f1_score(truth,pred,average='micro'))
        f1_macro.append(f1_score(truth,pred,average='macro'))
    return [  np.mean(val)  for val in  [acc,f1_micro,f1_macro] ]


def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    #stemming and lowercasing (no stopword removal
    words = [w for w in raw_text.lower().split()]
    return (" ".join(words))

def func_transformer(data): 
    encoder = {'bathroom':1,'bedroom':2,'dining room':3 ,'gym':4,'home':5,'indoor':6,'kitchen':7,'laundry':8,'living room':9,'other':10,'outside':11,'porch':12,'laundry':13}
    output =np.array([  encoder[e] for e in  data ] ).reshape(-1,1)
    return output  

def get_subj_func(): 
    subj =  {'subj':1,'obj':2}
    senti = pickle.load(open('senti_analyzer.pkl','rb') ) 
    def subj_func(data): 
        nonlocal subj 
        nonlocal senti
        out_list= list()
        for e in data: 
            out_list.append(subj[senti.classify(e)]) 
        return np.array(out_list).reshape(-1,1)
    return  subj_func

def extract_polarity(data): 
    dataset_list = []
    sid = SentimentIntensityAnalyzer() 
    for i,submission in enumerate(data): 
        patient_list = list() 
        neg,neu,pos = list(),list(),list()
        for sent in sent_tokenize(submission):  
            tmp = sid.polarity_scores(sent) 
            neg.append(tmp['neg'])
            neu.append(tmp['neu'])
            pos.append(tmp['pos'])
        dataset_list.append(pd.DataFrame(data={'neg':np.mean(neg),'neu':np.mean(neu),'pos':np.mean(pos)},index=[i])) 
    output = pd.concat(dataset_list).to_numpy() 
    return output 
