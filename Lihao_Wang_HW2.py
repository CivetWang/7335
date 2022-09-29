# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:30:41 2022

@author: harry
"""

# Homework 2
import numpy as np
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import itertools
from scipy import mean
import matplotlib.pyplot as plt

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function
# 3. Sample data from ML1 proj
inbound =  np.genfromtxt('C:/Users/harry/Desktop/MSDS/7335/cardio_train.csv', delimiter=';')
M = inbound[1:,1:-1]
L = inbound[1:,-1]
n_folds = 5

data = (M, L, n_folds)

clfsList = {RandomForestClassifier: 
            {'n_estimators':{1,4,8}
            ,'max_depth':{10,11,12}
            ,'min_samples_split':{5,10,15}
            }
        ,LogisticRegression:
            {'C':{1,3,5}
            ,'tol':{0.01,0.1}
        }
        ,DecisionTreeClassifier:
            {'criterion': ['gini', 'entropy']
             ,'max_depth': list(range(2,10,1))
             ,'min_samples_leaf': list(range(5,10,1))
            }
}

def grid(params):
    keys = params.keys()
    vals = params.values()
    for rowVals in itertools.product(*vals):
        yield dict(zip(keys, rowVals))

def run (a_clf, data, clf_hyper={}):
	M, L, n_folds = data 
	kf = KFold(n_splits=n_folds) 
	ret = {} 
	paramGrid = list(grid(clf_hyper)) 
	paramRes = [] 
	scaler = preprocessing.StandardScaler()
	for paramId, param in enumerate(paramGrid):
		ret = {} 
		for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
			clf = a_clf(**param) 
			Mtrain = scaler.fit_transform(M[train_index])
			Mtest = scaler.fit_transform(M[test_index])
			clf.fit(Mtrain, L[train_index])
			pred = clf.predict(Mtest)			
			ret[ids]= {'clf': clf  
                      ,'accuracy': accuracy_score(L[test_index], pred)
                      ,'param':param
                      ,'train_index': train_index
                      ,'test_index': test_index
                      }
		acc=  np.mean([r['accuracy'] for r in ret.values()])
		paramRes.append({'paramsId':paramId
                         ,'meanAccuracy':  acc
                         ,'params':param
                         ,'CV':ret
                         })
	return paramRes



allResults=[]
for clf, params in clfsList.items():
		results = run(a_clf=clf,data=data, clf_hyper=params) 
		bestAcc = max([r['meanAccuracy'] for r in results]) 
		bestAccId = ([r['paramsId'] for r in results if r['meanAccuracy']==bestAcc])[0] 
		allResults.append({'model':clf.__name__
                           ,'results':results 
                           ,'bestMeanAccuracy':bestAcc
                           ,'bestMeanAccurancId':bestAccId
                           })





plotacc=[]
plotaccmean=[]
plotY=[]
for clfs in allResults:
	for trial in clfs['results']:
		plotacc.append((list(cv['accuracy'] for cv in trial['CV'].values())))
		plotaccmean.append(trial['meanAccuracy'])
		plotY.append(clfs['model']  + ','.join("{!s}={!r}".format(key,val) for (key,val) in trial['params'].items()))
		
sortIdx = list(np.argsort(plotaccmean))

bestaccmean = [plotaccmean[i] for i in sortIdx]       
bestX = [plotacc[i] for i in sortIdx]       
bestY = [plotY[i] for i in sortIdx]       

print("Best Classifier"," ",bestY[-1]," with Mean Accuracy:",bestaccmean[-1])


fig, ax = plt.subplots(figsize=(15,len(bestaccmean)*.4))
fig.subplots_adjust(left=0.4,right=0.9,top=0.95,bottom=0.1)
ax.boxplot(bestX,vert=False,labels=bestY,showmeans=True) 





plt.savefig(fname = 'plot.png')

plt.show()