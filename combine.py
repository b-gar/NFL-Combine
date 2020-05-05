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
logreg = LogisticRegression(random_state = 24)
rf = RandomForestClassifier(random_state = 24)
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 3)

# Initialize Lists for Fold Accuracies
logacc = []
rfacc = []
gnbacc = []
knnacc = []

# Start K-Fold Cross-Validation
kf = KFold(n_splits = 10)
for k, (train, test) in enumerate(kf.split(X, Y)):
    
    # Logistic Regression
    print("\n")
    print("===== Logistic Regression =====")
    logreg.fit(X[train], Y[train])
    logpred = logreg.predict(X[test])
    logcm = confusion_matrix(Y[test], logpred) # tn,fp,fn,tp = logcm.ravel()
    print(logcm)
    loga = metrics.accuracy_score(Y[test], logpred)
    logacc.append(loga)
    print("Accuracy:", metrics.accuracy_score(Y[test], logpred))
    
    # Random Forest
    print("\n")
    print("===== Random Forest =====")
    rf.fit(X[train], Y[train])
    rfpred = rf.predict(X[test])
    rfcm = confusion_matrix(Y[test], rfpred) # tn,fp,fn,tp = rfcm.ravel()
    print(rfcm)
    rfa = metrics.accuracy_score(Y[test], rfpred)
    rfacc.append(rfa)
    print("Accuracy:", metrics.accuracy_score(Y[test], rfpred))
    
    # Naive Bayes
    print("\n")
    print("===== Naive Bayes =====")
    gnb.fit(X[train], Y[train])
    gnbpred = gnb.predict(X[test])
    gnbcm = confusion_matrix(Y[test], gnbpred) # tn,fp,fn,tp = logcm.ravel()
    print(gnbcm)
    gnba = metrics.accuracy_score(Y[test], gnbpred)
    gnbacc.append(gnba)
    print("Accuracy:", metrics.accuracy_score(Y[test], gnbpred))
    
    # KNN
    print("\n")
    print("===== K-Nearest Neighbors =====")
    knn.fit(X[train], Y[train])
    knnpred = knn.predict(X[test])
    knncm = confusion_matrix(Y[test], knnpred) # tn,fp,fn,tp = logcm.ravel()
    print(knncm)
    knna = metrics.accuracy_score(Y[test], knnpred)
    knnacc.append(knna)
    print("Accuracy:", metrics.accuracy_score(Y[test], knnpred))
    
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

# Prep Data for Tukey's
resultlist = np.asarray(logacc + rfacc + gnbacc + knnacc, dtype = np.float32)
testlist = np.repeat(np.array(["LogisticRegression", "RandomForest", "NaiveBayes", "KNN"]), 10)
resultdf = pd.DataFrame({"Model": testlist, "Accuracy": resultlist})
      
# Plot Model Comparison
sns.boxplot(x = resultdf['Model'], y = resultdf['Accuracy'], width = 0.4)
plt.title("Comparing Model Accuracies Over 10 Cross-Validation Folds")
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

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features ")
plt.show()