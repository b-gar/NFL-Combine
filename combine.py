# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:02:30 2020

@author: Owner
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics

combine10 = pd.read_csv("Files\combine2010.csv")
combine11 = pd.read_csv("Files\combine2011.csv")
combine12 = pd.read_csv("Files\combine2012.csv")
combine13 = pd.read_csv("Files\combine2013.csv")
combine14 = pd.read_csv("Files\combine2014.csv")
combine15 = pd.read_csv("Files\combine2015.csv")
combine16 = pd.read_csv("Files\combine2016.csv")
combine17 = pd.read_csv("Files\combine2017.csv")
combine18 = pd.read_csv("Files\combine2018.csv")
combine19 = pd.read_csv("Files\combine2019.csv")
combine20 = pd.read_csv("Files\combine2020.csv")


df = pd.concat([combine12,combine13,combine14,combine15,combine16,combine17,combine18,combine18,combine20])
df.columns = ['Player', 'Position', 'School', 'College', 'Height', 'Weight', '40Yard', 'Vertical', 'Bench', 'Broad', '3Cone', 'Shuttle', 'Drafted']
del df['College']

# Get Players Name
df[['Name', 'ID']] = df.Player.apply(lambda x: pd.Series(str(x).split("\\")))
del df['Player']

# Convert Height to Number
df[['Feet','Inches']] = df['Height'].str.split('-', 1, expand=True)
df.Feet = df.Feet.astype(int)
df.Inches = df.Inches.astype(int)
df['Height'] = (12 * df['Feet']) + df['Inches']

# Turn Drafted Binary
df['Drafted'] = df['Drafted'].notnull().astype(int)

# One Hot Encode
dfohe = pd.concat([pd.get_dummies(df['Position']), df], axis = 1)
dfohe = dfohe.drop(['Feet', 'Inches', 'Position', 'ID'], axis = 1)

# Normalize
normCols = ['Height', 'Weight', '40Yard', 'Vertical', 'Bench', 'Broad', '3Cone', 'Shuttle']
dfnorm = df[normCols]
dfnorm=(dfnorm-dfnorm.mean())/dfnorm.std()

# Get Final DF and Drop NA's
dfclean = pd.concat([dfohe.drop(normCols, axis=1), dfnorm], axis = 1)
dfclean = dfclean.dropna()

# Get Features
ints1 = np.arange(0,25,1)
ints2 = np.arange(28,36,1)
ints = np.concatenate((ints1,ints2), axis=0).tolist()

x = dfclean.iloc[:,ints]
X = x.values

# Get Target
y = dfclean.Drafted
Y = y.values

# Initialized Methods
logreg = LogisticRegression()
rf = RandomForestClassifier()
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 3)

# Start K-Fold Cross-Validation
logacc = []
rfacc = []
gnbacc = []
knnacc = []

kf = KFold(n_splits=10)
for k, (train, test) in enumerate(kf.split(X, Y)):
    
    # Logistic Regression
    print("===== Logistic Regression =====")
    logreg.fit(X[train], Y[train])
    logpred = logreg.predict(X[test])
    logcm = confusion_matrix(Y[test], logpred) # tn,fp,fn,tp = logcm.ravel()
    print(logcm)
    loga = metrics.accuracy_score(Y[test], logpred)
    logacc.append(loga)
    print("Accuracy:", metrics.accuracy_score(Y[test], logpred))
    
    # Random Forest
    print("===== Random Forest =====")
    rf.fit(X[train], Y[train])
    rfpred = rf.predict(X[test])
    rfcm = confusion_matrix(Y[test], rfpred) # tn,fp,fn,tp = rfcm.ravel()
    print(rfcm)
    rfa = metrics.accuracy_score(Y[test], rfpred)
    rfacc.append(rfa)
    print("Accuracy:", metrics.accuracy_score(Y[test], rfpred))
    
     # Naive Bayes
    print("===== Naive Bayes =====")
    gnb.fit(X[train], Y[train])
    gnbpred = gnb.predict(X[test])
    gnbcm = confusion_matrix(Y[test], gnbpred) # tn,fp,fn,tp = logcm.ravel()
    print(gnbcm)
    gnba = metrics.accuracy_score(Y[test], gnbpred)
    gnbacc.append(gnba)
    print("Accuracy:", metrics.accuracy_score(Y[test], gnbpred))
    
    # KNN
    print("===== K-Nearest Neighbors =====")
    knn.fit(X[train], Y[train])
    knnpred = knn.predict(X[test])
    knncm = confusion_matrix(Y[test], knnpred) # tn,fp,fn,tp = logcm.ravel()
    print(knncm)
    knna = metrics.accuracy_score(Y[test], knnpred)
    knnacc.append(knna)
    print("Accuracy:", metrics.accuracy_score(Y[test], knnpred))
    
print("==================================================================")

logmean = np.mean(logacc)
rfmean = np.mean(rfacc)
gnbmean = np.mean(gnbacc)
knnmean = np.mean(knnacc)

print("Logistic Regression Cross-Validation Mean: " + str(logmean.round(3)))
print("Random Forest Cross-Validation Mean: " + str(rfmean.round(3)))
print("Naive Bayes Cross-Validation Mean: " + str(gnbmean.round(3)))
print("KNN Cross-Validation Mean: " + str(knnmean.round(3)))

print("==================================================================")

if logmean > rfmean and logmean > gnbmean and logmean > knnmean:
    print("Logistic Regression Had the Highest Accuracy")
elif rfmean > logmean and logmean > gnbmean and rfmean > knnmean:
    print("Random Forest Had the Highest Accuracy")
elif knnmean > logmean and knnmean > rfmean and knnmean > gnbmean:
    print("KNN Had the Highest Accuracy")
else:
    print("Naive Bayes Had the Highest Accuracy")