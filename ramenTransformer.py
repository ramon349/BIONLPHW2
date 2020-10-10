from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np 

def __init__(self,vectorizer_max_features=None): 
    self.vectorizer = CountVectorizer()
    self.encoder = OrdinalEncoder() 
def fit(self,x,y=None): 
    self.vectorizer.fit(x['fall_description'])
    self.encoder.fit(x['simplified_location'])
def transform(self,x,y=None): 
    counts =self.vectorizer.transfomr(x['fall_description']).toArray()
    location = self.encoder.transform(x['simplified_location']).toArray(())
    return np.hstack((counts,location))
def fit_transform(self,x,y=None): 
    self.fit(x,y) 
    return self.transform(x,y)