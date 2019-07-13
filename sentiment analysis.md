# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 00:02:55 2019

@author: Sarthak
"""

**Loading Data :** 
```python
reviews_train = []
for line in open('C:\\Users\\Sarthak\\.spyder-py3\\movie_data\\full_train.txt', 'r',errors='ignore'):
    reviews_train.append(line.strip())
reviews_test = []
for line in open('C:\\Users\\Sarthak\\.spyder-py3\\movie_data\\full_test.txt', 'r',errors='ignore'):
    reviews_test.append(line.strip())
    
 ```
**Cleaning Data:**
```python

import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews
reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

```

**One-Hot encoding all the reviews using count-vectorizer:**
```python

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

```

**Building Classifier:**
```python

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

```

**Trying different values of the learning rate to check which one gives highest accuracy score:**
```python

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%d: %d" % (c, accuracy_score(y_val, lr.predict(X_val))))

```


