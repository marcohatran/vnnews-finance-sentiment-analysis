import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump, load
from sklearn.metrics import classification_report  

df = pd.read_csv("data.csv", encoding= "utf-8")
data = df.dropna()

sentences = data['tokenize_content']
X = []
for sentence in sentences:
  string = str(sentence)
  X.append(string)

y = np.array(data['price_trend'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf',  MultinomialNB()),])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}
clf = GridSearchCV(text_clf, tuned_parameters, cv=10 )
clf.fit(x_train, y_train)
dump(clf, 'model.pkl') 

print(classification_report(y_test, clf.predict(x_test), digits=4))