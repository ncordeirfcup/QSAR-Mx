import pandas as pd
from sklearn.linear_model import LinearRegression
from rm2 import rm2
from sklearn.metrics import r2_score
import numpy as np

class lco():
      def __init__(self,df,X,y):
            self.df=df
            self.X=X
            self.y=y
      def fit(self):
          model=LinearRegression()
          l1,l2=[],[]
          n=self.df.iloc[:,0:1]
          c1=self.df.iloc[:,1:2]
          c2=self.df.iloc[:,2:3]
          gr_1=self.df.groupby([c1.columns[0]]).count().sort_values(self.df.iloc[:,0:1].columns[0], ascending=False).reset_index()
          for i in range((len(gr_1[c1.columns[0]].unique()))):
              testi=pd.DataFrame(gr_1[c1.columns[0]].unique()).iloc[i]
              testi1=self.df[(self.df[c1.columns[0]]==testi.values[0])]
              traini=self.df.drop(testi1.index.values)
              #print(testi)
              yts=testi1[self.y.columns].reset_index().drop('index',axis=1)
              Xts=testi1[self.X.columns].reset_index().drop('index',axis=1)
              ytr=traini[self.y.columns].reset_index().drop('index',axis=1)
              Xtr=traini[self.X.columns].reset_index().drop('index',axis=1)
              model.fit(Xtr,ytr)
              ytspr=model.predict(Xts)
              tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
              l1.append(tsdf)
              tsc=pd.concat(l1,axis=0)
              tsc.columns=['Active','Predict']
          tsc['Aver']=tsc['Active'].values.mean()
          tsc['diff']=tsc['Active']-tsc['Predict']
          tsc['diff2']=tsc['Active']-tsc['Aver']
          q2_1=1-((tsc['diff']**2).sum()/(tsc['diff2']**2).sum())
          r2pr_1=r2_score(tsc['Active'],tsc['Predict'])
          rm2tr_1,drm2tr_1=rm2(tsc.iloc[:,0:1],tsc.iloc[:,1:2]).fit()
          q2_1,r2pr_1,rm2tr_1,drm2tr_1
          ###########secondcolumn#############
          gr_2=self.df.groupby([c2.columns[0]]).count().sort_values(self.df.iloc[:,0:1].columns[0], ascending=False).reset_index()
          for i in range((len(gr_2[c2.columns[0]].unique()))):
              testi=pd.DataFrame(gr_2[c2.columns[0]].unique()).iloc[i]
              testi1=self.df[(self.df[c2.columns[0]]==testi.values[0])]
              traini=self.df.drop(testi1.index.values)
              #print(testi)
              yts=testi1[self.y.columns].reset_index().drop('index',axis=1)
              Xts=testi1[self.X.columns].reset_index().drop('index',axis=1)
              ytr=traini[self.y.columns].reset_index().drop('index',axis=1)
              Xtr=traini[self.X.columns].reset_index().drop('index',axis=1)
              model.fit(Xtr,ytr)
              ytspr=model.predict(Xts)
              tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
              l2.append(tsdf)
              tsc=pd.concat(l2,axis=0)
              tsc.columns=['Active','Predict']
          tsc['Aver']=tsc['Active'].values.mean()
          tsc['diff']=tsc['Active']-tsc['Predict']
          tsc['diff2']=tsc['Active']-tsc['Aver']
          q2_2=1-((tsc['diff']**2).sum()/(tsc['diff2']**2).sum())
          r2pr_2=r2_score(tsc['Active'],tsc['Predict'])
          rm2tr_2,drm2tr_2=rm2(tsc.iloc[:,0:1],tsc.iloc[:,1:2]).fit()
          q2_2,r2pr_2,rm2tr_2,drm2tr_2
          ############Final############
          q2=(q2_1+q2_2)/2
          r2pr=(r2pr_2+r2pr_1)/2
          rm2tr=(rm2tr_2+rm2tr_1)/2
          drm2tr=(drm2tr_2+drm2tr_1)/2
          return q2,r2pr,rm2tr,drm2tr    