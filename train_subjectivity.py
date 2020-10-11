from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import * 
import pickle 

subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj') ]
train_subj_docs = subj_docs[:2500]
test_subj_docs = subj_docs[2500:]
train_obj_docs = obj_docs[:2500]
test_obj_docs = obj_docs[2500:]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set) 
with open('senti_analyzer.pkl','wb') as f: 
    pickle.dump(sentim_analyzer,f)
with open('classifier.pkl','wb') as f: 
    pickle.dump(classifier,f)