# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:03:03 2020

@author: ADMIN
"""

import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import statsmodels.formula.api as sm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

class classification():
    def __init__(self):
        self.df=pd.read_csv('D:\\datasets\\WA_Fn-UseC_-HR-Employee-Attrition.csv')
        
        
        
    def pre_process_data(self):
        le = preprocessing.LabelEncoder()
        for column_name in self.df.columns:
            if(type(self.df[column_name][0])==str):
                label=le.fit_transform(self.df[column_name])
                self.df[column_name]=label
            
    def attribute_selection(self):
        self.corr=self.df.corr()
        print(self.corr)
        wcorr = self.df.corr()
        plt.matshow(wcorr.abs())
        plt.colorbar()
        plt.xticks(range(len(wcorr.columns)), wcorr.columns, rotation='vertical');
        plt.yticks(range(len(wcorr.columns)), wcorr.columns);
        wcorr.abs().style.background_gradient()
        
        
    def logistic_regression(self):
        self.model = sm.logit('Attrition ~ Age + JobInvolvement + \
                         JobLevel + JobRole + JobSatisfaction + \
                         MonthlyIncome + OverTime +StockOptionLevel + \
                         TotalWorkingYears + YearsAtCompany + YearsInCurrentRole + \
                         YearsWithCurrManager ', self.df).fit()
        
        print(self.model.summary())
        pvalue=self.model.pvalues<0.05   
        print(pvalue)
        
        cmf = pd.DataFrame(self.model.pred_table())
        cmf.columns = ['Predicted_Class_0', 'Predicted_Class_1']
        cmf.index = ['True_Class_0', 'True_Class_1']
        acc=(cmf['Predicted_Class_0']['True_Class_0']+cmf['Predicted_Class_1']['True_Class_1'])/ \
        len(self.df['Attrition'])
        print(cmf)
        print(acc)
        
        
    def navie_bayes(self):
        self.features=zip(self.df['Age'],self.df['JobInvolvement'], \
                     self.df['JobSatisfaction'],self.df['StockOptionLevel'], \
                     self.df['YearsAtCompany'],self.df['YearsInCurrentRole'], \
                     self.df['YearsWithCurrManager'])
        feature_l=list(self.features)
        self.features=[]
        for feature in feature_l:
            self.features.append(list(feature))  
        y_true=self.df['Attrition']
        y_true=np.array(y_true)
        self.model = GaussianNB()
        self.model.fit(self.features,y_true)
        y_pred=[]
        for j in range(len(y_true)):
            y_pred.append(self.model.predict([self.features[j]]))
        cmf=confusion_matrix(y_true, y_pred)
        acc=accuracy_score(y_true,y_pred)
        cf=classification_report(y_true,y_pred)
        print(cmf)
        print(acc)
        print(cf)
 
    def KNN(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.features=zip(self.df['Age'],self.df['JobInvolvement'], \
                     self.df['JobSatisfaction'],self.df['StockOptionLevel'], \
                     self.df['YearsAtCompany'],self.df['YearsInCurrentRole'], \
                     self.df['YearsWithCurrManager'])
        feature_l=list(self.features)
        self.features=[]
        for feature in feature_l:
            self.features.append(list(feature))  
        y_true=self.df['Attrition']
        y_true=np.array(y_true)
        self.model.fit(self.features,y_true)
        y_pred=self.model.predict(self.features)
        cmf=confusion_matrix(y_true,y_pred)
        acc=accuracy_score(y_true,y_pred)
        cf=classification_report(y_true,y_pred)
        print(cmf)
        print(acc)
        print(cf)

       
    def svm(self):
        self.clf = svm.SVC(kernel='linear')
        self.features=zip(self.df['Age'],self.df['JobInvolvement'], \
                     self.df['JobSatisfaction'],self.df['StockOptionLevel'], \
                     self.df['YearsAtCompany'],self.df['YearsInCurrentRole'], \
                     self.df['YearsWithCurrManager'])
        feature_l=list(self.features)
        self.features=[]
        for feature in feature_l:
            self.features.append(list(feature))  
        y_true=self.df['Attrition']
        y_true=np.array(y_true)
        self.clf.fit(self.features,y_true)
        y_pred=self.clf.predict(self.features)
        cmf=confusion_matrix(y_true, y_pred)
        acc=accuracy_score(y_true,y_pred)
        cf=classification_report(y_true,y_pred)
        print(cmf)
        print(acc)
        print(cf)
        
    

        
model=classification()
model.pre_process_data()
#model.attribute_selection()
#model.logistic_regression()
#model.navie_bayes()
#model.KNN()
#model.svm()


#Accuracy of KNN > logistic_regression > navie_bayes > SVM


        
        
        
    
            

        
 
        
    
