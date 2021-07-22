import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly .offline as offline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.externals import joblib

#Importing dataset and examining it
dataset = pd.read_csv("spring.csv")
pd.set_option('display.max_columns', None) # Will ensure that all columns are displayed
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

#Converting Categorical features into Numerical features
def converter(column):
    if column >= 42.389417:
        return 1 # High pm2_5 reported
    else:
        return 0 # Low pm2_5 reported

dataset['pm2_5_reported'] = dataset['pm2_5_reported'].apply(converter)
print(dataset.head())

## Plotting Correlation Heatmap
#corrs = dataset.corr()
#figure = ff.create_annotated_heatmap(
#    z=corrs.values,
#    x=list(corrs.columns),
#    y=list(corrs.index),
#    annotation_text=corrs.round(2).values,
#    showscale=True)
#offline.plot(figure,filename='corrheatmap.html')

# Dividing dataset into label and feature sets
X = dataset.drop(['pm2_5_reported','station_id'], axis = 1) # Features
Y = dataset['pm2_5_reported'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)

# Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())


#########################################################################################################
# Tuning the AdaBoost parameter 'n_estimators' and implementing cross-validation using Grid Search
abc = AdaBoostClassifier(random_state=1)
grid_param = {'n_estimators': [5,10,20,30,40,50]}

gd_sr = GridSearchCV(estimator=abc, param_grid=grid_param, scoring='f1', cv=5)

#"""
#In the above GridSearchCV(), scoring parameter should be set as follows:
#scoring = 'accuracy' when you want to maximize prediction accuracy
#scoring = 'recall' when you want to minimize false negatives
#scoring = 'precision' when you want to minimize false positives
#scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
#"""

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

# Building AdaBoost using the tuned parameter
abc = AdaBoostClassifier(n_estimators=50, random_state=1)
abc.fit(X_train,Y_train)
featimp = pd.Series(abc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = abc.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])

#######################################################################################################

##Tuning the Gradient Boost parameter 'n_estimators' and implementing cross-validation using Grid Search
#gbc = GradientBoostingClassifier(random_state=1)
#grid_param = {'n_estimators': [10,20,30,40,50,60,70], 'max_depth' : [5,6,7,8,9,10,11,12], 'max_leaf_nodes': [8,12,16,20,24,28,32]}

#gd_sr = GridSearchCV(estimator=gbc, param_grid=grid_param, scoring='f1', cv=5)

##"""
##In the above GridSearchCV(), scoring parameter should be set as follows:
##scoring = 'accuracy' when you want to maximize prediction accuracy
##scoring = 'recall' when you want to minimize false negatives
##scoring = 'precision' when you want to minimize false positives
##scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
##"""

#gd_sr.fit(X_train, Y_train)

#best_parameters = gd_sr.best_params_
#print(best_parameters)

#best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
#print(best_result)

## Building Gradient Boost using the tuned parameter
#gbc = GradientBoostingClassifier(n_estimators=60, max_depth=9, max_leaf_nodes=32, random_state=1)
#gbc.fit(X_train,Y_train)
#featimp = pd.Series(gbc.feature_importances_, index=list(X)).sort_values(ascending=False)
#print(featimp)

#Y_pred = gbc.predict(X_test)
#print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

#conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(conf_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', conf_mat)
#print('TP: ', conf_mat[1,1])
#print('TN: ', conf_mat[0,0])
#print('FP: ', conf_mat[0,1])
#print('FN: ', conf_mat[1,0])

#####################################################################################################

## Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
#rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
#grid_param = {'n_estimators': [200, 250, 300, 350, 400, 450, 500]}

#gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring='precision', cv=5)

##"""
##In the above GridSearchCV(), scoring parameter should be set as follows:
##scoring = 'accuracy' when you want to maximize prediction accuracy
##scoring = 'recall' when you want to minimize false negatives
##scoring = 'precision' when you want to minimize false positives
##scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
##"""

#gd_sr.fit(X_train, Y_train)

#best_parameters = gd_sr.best_params_
#print(best_parameters)

#best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
#print(best_result)

## Building random forest using the tuned parameter
#rfc = RandomForestClassifier(n_estimators=450, criterion='entropy', max_features='auto', random_state=1)
#rfc.fit(X_train,Y_train)
#featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
#print(featimp)

#Y_pred = rfc.predict(X_test)
#print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

#conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(conf_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', conf_mat)
#print('TP: ', conf_mat[1,1])
#print('TN: ', conf_mat[0,0])
#print('FP: ', conf_mat[0,1])
#print('FN: ', conf_mat[1,0])   

## Selecting features with higher significance and building random forest
#X1 = dataset[['pm2_5_new_standard', 'temperature', 'latitude']]

##feature_scaler = StandardScaler()
#X1_scaled = feature_scaler.fit_transform(X1)
#X1_train, X1_test, Y1_train, Y1_test = train_test_split( X1_scaled, Y, test_size = 0.3, random_state = 100)

#smote = SMOTE(random_state = 101)
#X1_train,Y1_train = smote.fit_sample(X1_train,Y1_train)

##rfc = RandomForestClassifier(n_estimators=450, criterion='entropy', max_features='auto', random_state=1)
#rfc.fit(X1_train,Y1_train)

#Y_pred = rfc.predict(X1_test)
#print('Classification report: \n', metrics.classification_report(Y1_test, Y_pred))

#conf_mat = metrics.confusion_matrix(Y1_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(conf_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', conf_mat)
#print('TP: ', conf_mat[1,1])
#print('TN: ', conf_mat[0,0])
#print('FP: ', conf_mat[0,1])
#print('FN: ', conf_mat[1,0]) 






