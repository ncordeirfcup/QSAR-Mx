import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import numpy as np
#from testset_prediction import testset_prediction as tsp

class stepwise_selection():
    def __init__(self,X, y,cthreshold,vthreshold,max_steps,flot,forw,score,cvl):
        self.X=X
        self.y=y
        self.max_steps=max_steps
        self.cthreshold=cthreshold
        self.vthreshold=vthreshold
        self.flot=flot
        self.forw=forw
        self.score=score
        self.cvl=cvl
        initial_list=[]
        
    def correlation(self,X,cthreshold):
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] > float(cthreshold)) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
                    if colname in X.columns:
                        del X[colname] # deleting the column from the dataset
        return X   

    def variance(self,X,threshold):
        from sklearn.feature_selection import VarianceThreshold
        sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
        sel_var=sel.fit_transform(X)
        X=self.X[X.columns[sel.get_support(indices=True)]]    
        return X
    
    def fit_linear_reg(self,X,y):
        x=np.ones(X.shape[0])
        x=list(x)
        x=pd.DataFrame(x)
        x.columns=['constant']
        X=pd.concat([X,x],axis=1)
        dp=pd.concat([X,y],axis=1)
        table=MANOVA.from_formula('X.values~ y.values', data=dp).mv_test().results['y.values']['stat']
        Wilks_lambda=table.iloc[0,0]
        F_value=table.iloc[0,3]
        p_value=table.iloc[0,4]
        return Wilks_lambda,F_value,p_value,table
    
    def pretreat(self,X):
        X=self.correlation(X,self.cthreshold)
        X=self.variance(X,self.vthreshold)
        return X

    def feature_selection(self,X,y):
        mlr=LinearRegression()
        initial_list=[]
        included=list(initial_list)
        X=self.pretreat(X)
        sfs1 = sfs(mlr,k_features=self.max_steps,forward=self.forw,floating=self.flot,
               verbose=0,scoring=self.score,cv=self.cvl)
        sfs1 = sfs1.fit(X, y)
        a=list(sfs1.k_feature_names_)
        return a
        
    def fit_(self):
        mlr=LinearRegression()
        a=self.feature_selection(self.X,self.y)
        model=sm.OLS(self.y, sm.add_constant(pd.DataFrame(self.X[a])))
        results=model.fit()
        b=results.summary()
        return a,b