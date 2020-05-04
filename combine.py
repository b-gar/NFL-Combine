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
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
%matplotlib inline

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

# Start K-Fold Cross-Validation
logacc = []
rfacc = []
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

print("==================================================================")
logmean = np.mean(logacc)
rfmean = np.mean(rfacc)
print("Logistic Regression Cross-Validation Mean: " + str(logmean.round(3)))
print("Random Forest Cross-Validation Mean: " + str(rfmean.round(3)))
if logmean > rfmean:
    print("Logistic Regression Had a Higher Accuracy")
else:
    print("Random Forest Had a Higher Accuracy")