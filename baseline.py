from pyvi import ViTokenizer, ViPosTagger
import numpy as np
import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.multiclass import OneVsRestClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from joblib import dump, load
df = pd.read_csv("data.csv", encoding = "utf-8")
#print(df.head())


data = df.drop(['thread_url', 'thread_name','title?', 'created_date', 'sentiment'] , axis=1)
#print(data.head())


sents = data['content']
sents = sents.str.lower()
X = []
for sent in sents:
    split = ViTokenizer.tokenize( str(sent))
    #print (split)
    X.append(split)
#print(X)


labels = data['sentiment_final']
y = np.array(labels)
#print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf',  MultinomialNB()),])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

from sklearn.metrics import classification_report  
clf = GridSearchCV(text_clf, tuned_parameters, cv=10 )
clf.fit(x_train, y_train)
#dump(clf, 'model.pkl') 

print(classification_report(y_test, clf.predict(x_test), digits=4))