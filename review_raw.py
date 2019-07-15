#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
import re   
import nltk
nltk.download('stopwords')      
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

%matplotlib qt


#get the data
dataset = pd.read_csv('data/data_50000.csv',names=["RATING", "TITLE", "REVIEW"])
dataset = dataset.drop(['TITLE'],axis = 1)
dataset["Liked"] = dataset["RATING"].apply(lambda score: 1 if score >= 3 else 0)

corpus=[]
for k in range(0,50000,1):
    review = re.sub('[^a-zA-Z]'," ", dataset['REVIEW'][k] )       #removing exclamation marks and others 
    review = review.lower()
    review = review.split()                                       #split into indivisual words
    ps = PorterStemmer()
    review = [ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
    #ps.stem helps to stem the words like loving,loves ---> love
    review = ' '.join(review)
    corpus.append(review)

a = []
f = []
#BAG OF WORDS MODEL
tf = TfidfVectorizer(max_features = 1300)
x = tf.fit_transform(corpus).toarray()
#print(tf.get_feature_names())
y = dataset.iloc[:,2].values

#split into test and train
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 0)

#use GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

a_GNB = accuracies.mean()
f_GNB = f1_score(y_pred,y_test)
a.append(a_GNB)
f.append(a_GNB)

#use MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

a_MNB = accuracies.mean()
f_MNB = f1_score(y_pred,y_test)
a.append(a_MNB)
f.append(a_MNB)


#Use Logistic regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

a_logistic = accuracies.mean()
f_logistic = f1_score(y_pred,y_test)
a.append(a_logistic)
f.append(f_logistic)

#DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

a_Decision_Tree = accuracies.mean()
f_Decision_Tree = f1_score(y_pred,y_test)
a.append(a_Decision_Tree)
f.append(f_Decision_Tree)

# data to plot
n_groups = 4

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, a, bar_width,
alpha=opacity,
color='b',
label='Accuracies')

rects2 = plt.bar(index + bar_width, f, bar_width,
alpha=opacity,
color='g',
label='F1_score')

plt.ylabel('Scores')
plt.title('Evaluation')
plt.xticks(index + bar_width, ('GaussianNB', 'MultinomialNB', 'Logistic_Regression', 'Decision Tree'))
plt.legend()

plt.tight_layout()
plt.show()


'''
#use SVC
classifier = SVC(kernel='poly',degree = 3)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

a_SVC = accuracies.mean()
f_SVC = f1_score(y_pred,y_test)
'''
