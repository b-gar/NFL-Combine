# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:02:30 2020

@author: Ben Garski
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
import pickle
from scipy.stats import levene
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Import Files
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

# Join Into One DF
df = pd.concat([combine12,combine13,combine14,combine15,combine16,combine17,combine18,combine18,combine20])

# Rename Columns
df.columns = ['Player', 'Position', 'School', 'College', 'Height', 'Weight', '40Yard', 'Vertical', 'Bench', 'Broad', '3Cone', 'Shuttle', 'Drafted']

# Drop College Column and Duplicates
del df['College']
df = df.drop_duplicates(subset = "Player")

# Check Position Levels
df.Position.value_counts()

# Remove DB and NT for Lack of Data
df = df[df.Position != 'DB']
df = df[df.Position != 'NT']

# Get Number and Percent of Missing Values by Column
df.isnull().sum().sort_values(ascending=False)
(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

# Check Descriptive Stats and Correlation Matrix
df.describe()
plt.figure(figsize=(15,6))
sns.heatmap(df.corr(), annot=True, square=True)

# Check Unique Value Counts by Column
df.apply(pd.Series.nunique).sort_values(ascending=False)

# Check Position Numbers Before Imputation
df.groupby('Position').count()
df.groupby('Position').median()

# Kickers and Punters Missing Most of the Testing Data
df = df[df.Position != 'K']
df = df[df.Position != 'P']

# Drop Athletes who Missed More than 2 Tests
df = df.dropna(subset = ['40Yard', 'Vertical', 'Bench', 'Broad', '3Cone', 'Shuttle'], thresh = 4)

# Imputate NA's of Position Group Medians
df['40Yard'] = df['40Yard'].fillna(df.groupby('Position')['40Yard'].transform('median'))
df['Vertical'] = df['Vertical'].fillna(df.groupby('Position')['Vertical'].transform('median'))
df['Bench'] = df['Bench'].fillna(df.groupby('Position')['Bench'].transform('median'))
df['Broad'] = df['Broad'].fillna(df.groupby('Position')['Broad'].transform('median'))
df['3Cone'] = df['3Cone'].fillna(df.groupby('Position')['3Cone'].transform('median'))
df['Shuttle'] = df['Shuttle'].fillna(df.groupby('Position')['Shuttle'].transform('median'))

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

# Get Numerical Columns
normCols = ['Height', 'Weight', '40Yard', 'Vertical', 'Bench', 'Broad', '3Cone', 'Shuttle']
dfnorm = df[normCols]

# Normalize
dfnorm = (dfnorm-dfnorm.mean())/dfnorm.std()

# Join DF's from Above
dfclean = pd.concat([dfohe.drop(normCols, axis=1), dfnorm], axis = 1)

# Get Features
x = dfclean.drop(['School', 'Name', 'Drafted'], axis = 1)
X = x.values

# Get Target
y = dfclean.Drafted
Y = y.values

# Initialized Methods
logreg = LogisticRegression(random_state = 24)
rf = RandomForestClassifier(random_state = 24)
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 3)

# Initialize Lists for Fold Accuracies
logacc = []
rfacc = []
gnbacc = []
knnacc = []

# Initialize Lists for Fold FPR
logfpr = []
rffpr = []
gnbfpr = []
knnfpr = []

# Start K-Fold Cross-Validation
kf = KFold(n_splits = 10)
for k, (train, test) in enumerate(kf.split(X, Y)):
    
    # Logistic Regression
    print("\n")
    print("===== Logistic Regression =====")
    logreg.fit(X[train], Y[train])
    logpred = logreg.predict(X[test])
    logcm = confusion_matrix(Y[test], logpred)
    logtn,logfp,logfn,logtp = logcm.ravel()
    print("Sensitivity: " + str(round(logtp/(logtp+logfn),2)))
    print("Specificity: " + str(round(logtn/(logtn+logfp),2)))
    print("False Negative Rate: " + str(round(logfn/(logfn+logtp),2)))
    print("False Positive Rate: " + str(round(logfp/(logfp+logtn),2)))
    print(logcm)
    logf = logfp/(logfp+logtn)
    loga = metrics.accuracy_score(Y[test], logpred)
    logfpr.append(logf)
    logacc.append(loga)
    print("Accuracy:", round(metrics.accuracy_score(Y[test], logpred),2))
    
    # Random Forest
    print("\n")
    print("===== Random Forest =====")
    rf.fit(X[train], Y[train])
    rfpred = rf.predict(X[test])
    rfcm = confusion_matrix(Y[test], rfpred)
    rftn,rffp,rffn,rftp = rfcm.ravel()
    print("Sensitivity: " + str(round(rftp/(rftp+rffn),2)))
    print("Specificity: " + str(round(rftn/(rftn+rffp),2)))
    print("False Negative Rate: " + str(round(rffn/(rffn+rftp),2)))
    print("False Positive Rate: " + str(round(rffp/(rffp+rftn),2)))
    print(rfcm)
    rff = rffp/(rffp+rftn)
    rfa = metrics.accuracy_score(Y[test], rfpred)
    rffpr.append(rff)
    rfacc.append(rfa)
    print("Accuracy:", round(metrics.accuracy_score(Y[test], rfpred),2))
    
    # Naive Bayes
    print("\n")
    print("===== Naive Bayes =====")
    gnb.fit(X[train], Y[train])
    gnbpred = gnb.predict(X[test])
    gnbcm = confusion_matrix(Y[test], gnbpred)
    gnbtn,gnbfp,gnbfn,gnbtp = gnbcm.ravel()
    print("Sensitivity: " + str(round(gnbtp/(gnbtp+gnbfn),2)))
    print("Specificity: " + str(round(gnbtn/(gnbtn+gnbfp),2)))
    print("False Negative Rate: " + str(round(gnbfn/(gnbfn+gnbtp),2)))
    print("False Positive Rate: " + str(round(gnbfp/(gnbfp+gnbtn),2)))
    print(gnbcm)
    gnbf = gnbfp/(gnbfp+gnbtn)
    gnba = metrics.accuracy_score(Y[test], gnbpred)
    gnbfpr.append(gnbf)
    gnbacc.append(gnba)
    print("Accuracy:", round(metrics.accuracy_score(Y[test], gnbpred),2))
    
    # KNN
    print("\n")
    print("===== K-Nearest Neighbors =====")
    knn.fit(X[train], Y[train])
    knnpred = knn.predict(X[test])
    knncm = confusion_matrix(Y[test], knnpred)
    knntn,knnfp,knnfn,knntp = knncm.ravel()
    print("Sensitivity: " + str(round(knntp/(knntp+knnfn),2)))
    print("Specificity: " + str(round(knntn/(knntn+knnfp),2)))
    print("False Negative Rate: " + str(round(knnfn/(knnfn+knntp),2)))
    print("False Positive Rate: " + str(round(knnfp/(knnfp+knntn),2)))
    print(knncm)
    knnf = knnfp/(knnfp+knntn)
    knna = metrics.accuracy_score(Y[test], knnpred)
    knnfpr.append(knnf)
    knnacc.append(knna)
    print("Accuracy:", round(metrics.accuracy_score(Y[test], knnpred),2))
    
# Save Models
logfile = "logisticRegression.pkl"
with open(logfile, 'wb') as file:
    pickle.dump(logreg, file)
    
rffile = "randomForest.pkl"
with open(rffile, 'wb') as file:
    pickle.dump(rf, file)

nbfile = "naiveBayes.pkl"
with open(nbfile, 'wb') as file:
    pickle.dump(gnb, file)

knnfile = "knn.pkl"
with open(knnfile, 'wb') as file:
    pickle.dump(knn, file)
    
print("==================================================================")

# Display the Mean Accuracy From Cross-Validation
logmean = np.mean(logacc)
rfmean = np.mean(rfacc)
gnbmean = np.mean(gnbacc)
knnmean = np.mean(knnacc)

print("Logistic Regression Cross-Validation Mean: " + str(logmean.round(3)))
print("Random Forest Cross-Validation Mean: " + str(rfmean.round(3)))
print("Naive Bayes Cross-Validation Mean: " + str(gnbmean.round(3)))
print("KNN Cross-Validation Mean: " + str(knnmean.round(3)))

# Dataframe for Accuracy Comparison & Tukey's Test
resultlist = np.asarray(logacc + rfacc + gnbacc + knnacc, dtype = np.float32)
testlist = np.repeat(np.array(["LogisticRegression", "RandomForest", "NaiveBayes", "KNN"]), 10)
resultdf = pd.DataFrame({"Model": testlist, "Accuracy": resultlist})
    
# Dataframe for FPR Comparison  
resultlist2 = np.asarray(logfpr + rffpr + gnbfpr + knnfpr, dtype = np.float32)
testlist2 = np.repeat(np.array(["LogisticRegression", "RandomForest", "NaiveBayes", "KNN"]), 10)
resultdf2 = pd.DataFrame({"Model": testlist2, "FPR": resultlist2})

# Plot Model Comparison for Accuracies
sns.boxplot(x = resultdf['Model'], y = resultdf['Accuracy'], width = 0.4)
plt.title("Comparing Model Accuracies Over 10 Cross-Validation Folds")
plt.show()

# Plot Model Comparison for FPR
plt.figure(figsize=(8,5))
sns.set_context("notebook", rc = {'axes.labelsize':18,'axes.titlesize':20})
sns.boxplot(x = resultdf2['Model'], y = resultdf2['FPR'], width = 0.4)
sns.despine()
plt.title("Comparing Model FPR Over 10 Cross-Validation Folds")
plt.xlabel('')
plt.xticks(rotation=45)
plt.show()

# Check for a Violation of ANOVA - Homogeneity of Variance
levenep = levene(logacc, rfacc, gnbacc, knnacc)[1]

# Run ANOVA if No Violation
if levenep > 0.05:
    anova = f_oneway(logacc, rfacc, gnbacc, knnacc)[1]
    
    # Run Tukey Test if Significant to Find Which Group Differs
    if anova < 0.05:
      
        # Tukey
        print(pairwise_tukeyhsd(resultdf['Accuracy'], resultdf['Model']))


print("==================================================================")

# Display Most Important Features for Random Forest
feature_imp = pd.Series(rf.feature_importances_, index = x.columns).sort_values(ascending = False)

plt.figure(figsize=(12,8))
sns.barplot(x = feature_imp, y = feature_imp.index)
sns.despine()
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features ")
plt.xticks(rotation=45)
plt.show()