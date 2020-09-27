###Last updated on 27/09/2020 
###This version includes 1:5 ratio of descriptors/trainset samples
###Addition includes MAE values in test set
###Save the descriptor+dependent parameter files
###Warning:Any na will be converted to 0

import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import os
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
#from kmca import kmca
from sequential_selection import stepwise_selection as sq
from loo import loo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from applicability import apdom
import math
import numpy as np
import threading
from lco import lco


form = tk.Tk()
form.title("QSAR-mx")
form.geometry("650x580")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)
#tab2 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="Module-2: Grid search based selection")
#tab_parent.add(tab2, text="Model development")

initialdir=os.getcwd()

reg=LinearRegression()

def data1():
    filename = askopenfilename(initialdir=initialdir,title = "Select Descriptor Data")
    firstEntryTabOne.delete(0, END)
    firstEntryTabOne.insert(0, filename)
    global a_
    a_,b_=os.path.splitext(filename)
    global file
    file = pd.read_csv(filename)
    return file


def variance(X,threshold):
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X

def corr(df):
    lt=[]
    df1=df.iloc[:,1:]
    for i in range(len(df1)):
        x=df1.values[i]
        x = sorted(x)[0:-1]
        lt.append(x)
    flat_list = [item for sublist in lt for item in sublist]
    return max(flat_list),min(flat_list)


    
def method1(x1,x2,f1,f2):
    x1=x1.fillna(0)
    x2=x2.fillna(0)
    l=[]
    for i in x1:
        for j in f1:
            for k in f2:
                x=x1[i]*f1[j]+x2[i]*f2[k]
                l.append(x)
    l=pd.DataFrame(l).transpose()
    l.columns=x1.columns
    l=l.add_suffix('_dpmix')
    return l

def method2(x1,x2,f1,f2):
    x1=x1.fillna(0)
    x2=x2.fillna(0)
    l=[]
    for i in x1:
        for j in f1:
            for k in f2:
                x=abs(x1[i]*f1[j]-x2[i]*f2[k])
                l.append(x)
    l=pd.DataFrame(l).transpose()
    l.columns=x1.columns
    l=l.add_suffix('_dnmix')
    return l

def check_data(dct,n):
    ds=file.iloc[:,0:n+4]
    filex1=file.drop_duplicates(subset=list(ds.columns)[1::])
    fdupl=file.drop(filex1.index.values)      
    if fdupl.shape[0]>0:
       messagebox.showinfo('Warning','Duplicate samples found in the data, will be removed')
       #print(str(directory))
       tbn='duplicates2.csv'
       fdupl.to_csv(os.path.join(dct,tbn), index=False)
       #fdupl.to_csv(str(a_)+'duplicates.csv', index=False)
       tbn2= 'duplicates_removed.csv'
       filex1.to_csv(os.path.join(dct,tbn2), index=False)
    else:
       filex1=file
    return filex1
    

def pointout(df,c1,c2,sp,n):
    #pc=int(100/n)
    gr1=df.groupby([c1.columns[0],c2.columns[0]]).count()[df.iloc[:,0:1].columns[0]].sort_values(ascending=False).reset_index().rename(columns={0:'count'})
    lsts2,lstr2=[],[]
    for i,j in gr1[[c1.columns[0],c2.columns[0]]].values:
        df1=df[(df[c1.columns[0]]==i) & (df[c2.columns[0]]==j)]
        dfts=df1.iloc[sp::n,:]
        dftr=df1.drop(dfts.index.values)
        lsts2.append(dfts)
        lstr2.append(dftr)
        dftrp=pd.concat(lstr2,axis=0)
        dftsp=pd.concat(lsts2,axis=0)
    return dftrp,dftsp

def mixtureout(df,c1,c2,sp,n):
    #pc=int(100/n)
    gr1=df.groupby([c1.columns[0],c2.columns[0]]).count()[df.iloc[:,0:1].columns[0]].sort_values(ascending=False).reset_index().rename(columns={0:'count'})
    ts=gr1.iloc[sp::n, :]
    tr=gr1.drop(ts.index.values)
    #print(tr[[c1.columns[0],c2.columns[0]]].values)    
    ls,lst=[],[]
    for i,j in tr[[c1.columns[0],c2.columns[0]]].values:
        df1=df[(df[c1.columns[0]]==i) & (df[c2.columns[0]]==j)]
        ls.append(df1)
        dftrm=pd.concat(ls,axis=0)

    for i,j in ts[[c1.columns[0],c2.columns[0]]].values:
        df1=df[(df[c1.columns[0]]==i) & (df[c2.columns[0]]==j)]
        lst.append(df1)
        dftsm=pd.concat(lst,axis=0)
    return dftrm,dftsm

def compout(df,c1,c2,sp,n,directory):
    #pc=int(100/n)
    gr_1=df.groupby([ c1.columns[0]]).count().sort_values(df.iloc[:,0:1].columns[0], ascending=False).reset_index()
    gr_1[ c1.columns[0]].unique() ####get the list of HBD present
    test=pd.DataFrame(gr_1[ c1.columns[0]].unique()).iloc[sp::(2*n), :] ####divide the HBD to test set
    train=pd.DataFrame(gr_1[ c1.columns[0]].unique()).drop(test.index.values) ####divide the HBD to training set
    nl1,nl2=[],[]
    for i in train.values:
        df1=df[(df[ c1.columns[0]]==i[0])]
        nl1.append(df1)
        dftra=pd.concat(nl1,axis=0)
    for i in test.values:
        df1=df[(df[ c1.columns[0]]==i[0])]
        nl2.append(df1)
        dftsa=pd.concat(nl2,axis=0)
    #dftra.shape, dftsa.shape
    gr_2=dftsa.groupby([ c2.columns[0]]).count().sort_values(dftra.iloc[:,0:1].columns[0], ascending=False).reset_index()
    gr_2[ c2.columns[0]].unique() ####get the list of Salt present
    test2=pd.DataFrame(gr_2[ c2.columns[0]].unique()).iloc[sp::(2*n), :] ####divide the HBD to test set
    train2=pd.DataFrame(gr_2[ c2.columns[0]].unique()).drop(test2.index.values) ####divide the HBD to training set0
    nl3,nl4=[],[]
    for i in train2.values:
        df2=dftra[(dftra[ c2.columns[0]]==i[0])]
        nl3.append(df2)
        dftra2=pd.concat(nl3,axis=0)
    for i in test2.values:
        df2=dftra[(dftra[ c2.columns[0]]==i[0])]
        nl4.append(df2)
        dftsa2=pd.concat(nl4,axis=0)
    if len(nl4)==0:
        fdftsa=pd.concat([dftsa],axis=0)
        fdftra=df.drop(fdftsa.index.values)
        #messagebox.showinfo('Warning','No second element found in seed'+str(sp)+' and interval '+str(n))
        filew = 'Warning.txt'
        file_path = os.path.join(directory, filew)
        filew = open(file_path, "a")
        filew.write('No second element found in seed'+str(sp)+' and interval '+str(n)+'\n')
        filew.close()
    else:
        fdftsa=pd.concat([dftsa,dftsa2],axis=0)
        fdftra=df.drop(fdftsa.index.values)
    
    return fdftra,fdftsa

def datasel(i,j,filex1,dct):
    n=int(thirdEntryTabThreer3c2_1.get())
    #filex1=check_data(n)
    sn=filex1.iloc[:,0:1]
    inc=filex1.iloc[:,0:n+5]
    #inc2=filex1.iloc[:,1:n+4]
    inc3=file.iloc[:,1:5]
    inc4=file.iloc[:,3:5]
    d=filex1.iloc[:,n+5:]
    c1=filex1.iloc[:,1:2]
    c2=filex1.iloc[:,2:3]
    f1=filex1.iloc[:,3:4]
    f2=filex1.iloc[:,4:5]
    ds=d.shape[1]
    x1=d.iloc[:,0:int(ds/2)]
    x2=d.iloc[:,int(ds/2):]
    x2.columns=x1.columns
    if var1.get():
       l=method1(x1,x2,f1,f2)
       var=float(fourthEntryTabThreer5c2_1.get())
       l=variance(l,var)
    elif var2.get():
       l1=method1(x1,x2,f1,f2)
       l2=method2(x1,x2,f1,f2)
       l=pd.concat([l1,l2],axis=1)
       var=float(fourthEntryTabThreer5c2_1.get())
       l=variance(l,var)
    fileam=pd.concat([inc,l],axis=1)
    #fileam.to_csv('fileam.csv',index=False)
    
    #print(ls1,ls2)
    
    if Selection.get()=='Point-out':
        trp,tsp=pointout(fileam,c1,c2,i,j)
        method='po'
    elif Selection.get()=='Mixture-out':
        trp,tsp=mixtureout(fileam,c1,c2,i,j)
        method='mo'
    elif Selection.get()=='Compound-out':
         trp,tsp=compout(fileam,c1,c2,i,j,dct)
         method='co'
    seed=i
    interval=j
    trdp=trp.drop(inc3.columns,axis=1)
    tsdp=tsp.drop(inc3.columns,axis=1)
    trdp2=trp.drop(inc4.columns,axis=1)
    trsp=trp[inc.iloc[:,0:5].columns]
    trsp['Set']='Train'
    tssp=tsp[inc.iloc[:,0:5].columns]
    tssp['Set']='Test'
    tsp=pd.concat([trsp,tssp],axis=0)
    name=str(method)+str(seed)+str(interval)+'_division.csv'
    tsp.to_csv(os.path.join(dct,name),index=False)
    ###
    sc=trdp.iloc[:,1:n]
    trdp=trdp.drop(sc,axis=1)
    trdp=pd.concat([trdp,sc],axis=1)
    ###
    sc2=tsdp.iloc[:,1:n]
    tsdp=tsdp.drop(sc2,axis=1)
    tsdp=pd.concat([tsdp,sc2],axis=1)
    ###
    #savename1= str(method)+str(seed)+str(interval)+'_tr.csv'
    #trdp.to_csv(savename1,index=False)
    #savename2 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save testset file")
    #savename2= str(method)+str(seed)+str(interval)+'_ts.csv'
    #tsdp.to_csv(savename2,index=False)
    return trdp,tsdp,trdp2
  
def wait_end(label, tk_var_end, num=0):
    label["text"] = "Processing " + " ." * num
    num += 1
    if num == 4:
        num = 0
    if not tk_var_end.get():
        form.after(500, wait_end, label, tk_var_end, num)


def execute():
    #b6_x.destroy()
    tk_process_lbl = tk.Label(form,font=('Helvetica 12 bold'),fg="red")
    tk_process_lbl.pack()
    tk_process_lbl.place(x=420,y=525)

    tk_var_end = tk.BooleanVar(False)
    wait_end(tk_process_lbl, tk_var_end)
    process = threading.Thread(
        target=seliter,
        kwargs=(dict(callback=lambda: tk_var_end.set(True)))
    )
    process.start()

    form.wait_variable(tk_var_end)
    form.after(500, tk_process_lbl.config, dict(text='Process completed',font=('Helvetica 12 bold'), fg="red"))
    #form.after(1500, form.quit)


def seliter(callback):
    directory=str(thirdEntryTabThreer3c2_2.get())
    if not os.path.isdir(directory):
       os.mkdir(directory)
    s1,i1,l1,l2,l3,l4,l5,l6,l7,l8=[],[],[],[],[],[],[],[],[],[]
    filex1=check_data(directory,int(thirdEntryTabThreer3c2_1.get()))
    for i in range(1,int(secondEntryTabOne.get())):
            for j in range(1,int(secondEntryTabOne_y.get())):
                file1,file2,file3=datasel(i,j,filex1,directory)         ######Entry
                ratio=100*(file2.shape[0]/(file2.shape[0]+file1.shape[0]))
                #print(ratio)
                cfilex="cumulative_results.txt"
                file_path2 = os.path.join(directory, cfilex)
                cfile = open(file_path2, "a")
                if file1.shape[0]>file2.shape[0]:
                   if ratio>15:
                      filex = str(i)+str(j)+"_results.txt"
                      file_path = os.path.join(directory, filex)
                      filer = open(file_path, "w")
                      o1,o2,o3,o4,o5,o6,o7,o8=process(file1,file2,file3,filer,cfile,i,j,directory)
                      s1.append(i)
                      i1.append(j)
                      l1.append(o1)
                      l2.append(o2)
                      l3.append(o3)
                      l4.append(o4)
                      l5.append(o5)
                      l6.append(o6)
                      l7.append(o7)
                      l8.append(o8)
                      filer.close()
                   else:
                      cfile.write('Seed '+str(i)+' and interval '+str(j)+' skipped as test set larger than training set'+"\n")
                      continue
                else:
                   cfile.write('Seed '+str(i)+' and interval '+str(j)+' skipped as test set larger than training set'+"\n")
                   continue
    
    Dict=dict([('seed', s1),('interval', i1),('ntr', l1), ('Q2LOO', l2),('Q2LCO',l3),('MAE_loo',l4),('nts', l5),('R2Pr', l6),('MAE_test', l7),('Max_intercorr', l8)]) 
    table=pd.DataFrame(Dict)
    tbname='Results_table.csv'
    table.to_csv(os.path.join(directory,tbname),index=False)       
    cfile.close()
    callback()

def process(file1,file2,file3,filer,cfile,i,j,dct):
    ls1,ls2,ls3,ls4,ls5,ls6,ls7,ls8=[],[],[],[],[],[],[],[]
    Xtr=file1.iloc[:,2:].reset_index().drop('index',axis=1)
    ytr=file1.iloc[:,1:2].reset_index().drop('index',axis=1)
    ntr=file1.iloc[:,0:1].reset_index().drop('index',axis=1)
    a,b,c,m,mx,mn,l,filer=trainsetfit2(Xtr,ytr,filer,file1) #####Entry
    mc=max(abs(mx),abs(mn))
    
    reg.fit(Xtr[a],ytr)
    r2=reg.score(Xtr[a],ytr)
    ypr=pd.DataFrame(reg.predict(Xtr[a]))
    ypr.columns=['Pred']
    rm2tr,drm2tr=rm2(ytr,l).fit()
    
    trdf=pd.concat([ytr,pd.DataFrame(ypr)],axis=1)
    trdf.columns=['Active','Predict']
    trdf['diff']=abs(trdf['Active']-trdf['Predict'])
    aard=(100*(trdf['diff']/trdf['Active'])).sum()/(ytr.shape[0])

    d=mean_absolute_error(ytr,ypr)
    e=(mean_squared_error(ytr,ypr))**0.5
    dloo=mean_absolute_error(ytr,l)
    eloo=(mean_squared_error(ytr,l))**0.5
    adstr=apdom(Xtr[a],Xtr[a])
    yadstr=adstr.fit() 
    dfl=pd.concat([ntr,Xtr[a],ytr],axis=1)
    df=pd.concat([ntr,Xtr[a],ytr,ypr,l,yadstr],axis=1)
    dftrm2=df[df['Outlier_info(standardization_approach)']=='Outlier']
    
    trname=str(i)+str(j)+'_trpred.csv'
    dfl.to_csv(os.path.join(dct,trname),index=False)
    
    lc=lco(file3,Xtr[a],ytr)
    lcoq2,lcorpred,lcorm2,lcodrm2=lc.fit()
    
    filer.write("Sub-training set results "+"\n")
    filer.write("System information "+"\n")
    filer.write("Seed value "+str(i)+"\n")
    filer.write("Interval value "+str(j)+"\n")
    filer.write('Dependent parameter'+str(ytr.columns[0])+"\n")
    filer.write("\n")
    filer.write("Selected features are:"+str(a)+"\n")    
    filer.write("Statistics:"+str(b)+"\n")
    filer.write('Training set results: '+"\n")
    filer.write('Number of observations '+str(Xtr.shape[0])+"\n")
    filer.write('No of outliers in training set: '+str(dftrm2.shape[0])+"\n")  
    filer.write('Maxmimum intercorrelation between descriptors: '+str(mc)+"\n")
    #filer.write('Minimum intercorrelation between descriptors: '+str(mn)+"\n")
    filer.write('MAE_fit: '+str(d)+"\n")
    filer.write('MAE_loo: '+str(dloo)+"\n")
    filer.write('RMSE_fit: '+str(e)+"\n")
    filer.write('RMSE_loo: '+str(eloo)+"\n")
    filer.write('Q2LOO: '+str(c)+"\n")
    filer.write('AARD: '+str(aard)+"\n")
    
    
    cfile.write("Sub-training set results "+"\n")
    cfile.write("System information "+"\n")
    cfile.write("Seed value "+str(i)+"\n")
    cfile.write("Interval value "+str(j)+"\n")
    cfile.write('Dependent parameter'+str(ytr.columns[0])+"\n")
    cfile.write("\n")
    cfile.write("Selected features are:"+str(a)+"\n") 
    cfile.write("Number of descriptors: "+str(len(a))+"\n")
    cfile.write('Training set results: '+"\n")
    cfile.write('Number of observations '+str(Xtr.shape[0])+"\n")
    cfile.write('No of outliers in training set: '+str(dftrm2.shape[0])+"\n")
    cfile.write('Maxmimum intercorrelation between descriptors: '+str(mc)+"\n")
    #cfile.write('Minimum intercorrelation between descriptors: '+str(mn)+"\n")
    cfile.write('Q2LOO: '+str(c)+"\n")
    cfile.write('Q2LCO: '+str(lcoq2)+"\n")
    cfile.write('MAE_fit: '+str(d)+"\n")
    cfile.write('MAE_loo: '+str(dloo)+"\n")
    cfile.write('AARD: '+str(aard)+"\n")
    
    
    if ytr.columns[0] in file2.columns:
       #Xts1=file2.iloc[:,3:]
       #Xts2=file2.iloc[:,1:2]
       #Xts=pd.concat([Xts1,Xts2],axis=1)
       Xts=file2.iloc[:,2:].reset_index().drop('index',axis=1)
       Xts=Xts.reset_index().drop('index',axis=1)
       #Xtr=file1.iloc[:,2:].reset_index().drop('index',axis=1)
       yts=file2.iloc[:,1:2].reset_index().drop('index',axis=1)
       nts=file2.iloc[:,0:1].reset_index().drop('index',axis=1)
       ytspr=pd.DataFrame(reg.predict(Xts[a]))
       ytspr.columns=['Pred']
       dts=mean_absolute_error(yts,ytspr)
       rm2ts,drm2ts=rm2(yts,ytspr).fit()
       aardts=100*(abs((yts-ytspr)/yts))/(yts.shape[0])
       tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
       tsdf.columns=['Active','Predict']
       tsdf['Aver']=m
       tsdf['Aver2']=tsdf['Predict'].mean()
       tsdf['diff']=tsdf['Active']-tsdf['Predict']
       tsdf['diff2']=tsdf['Active']-tsdf['Aver']
       tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
       r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
       r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
       aardts=(100*(abs(tsdf['diff']/tsdf['Active']))).sum()/(yts.shape[0])
       RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
       adts=apdom(Xts[a],Xtr[a])
       yadts=adts.fit()
       dftsm=pd.concat([nts,Xts[a],yts,ytspr,yadts],axis=1)
       dftsm2=dftsm[dftsm['Outlier_info(standardization_approach)']=='Outlier']
       filer.write('No of outliers in test set: '+str(dftsm2.shape[0])+"\n")
       dfts=pd.concat([nts,Xts[a],yts],axis=1)
       tsname=str(i)+str(j)+'_tspred.csv'
       dfts.to_csv(os.path.join(dct,tsname),index=False)
       filer.write('rm2LOO: '+str(rm2tr)+"\n")
       filer.write('delta rm2LOO: '+str(drm2tr)+"\n")
       filer.write("\n")
       filer.write('Test set results: '+"\n")
       filer.write('Number of observations '+str(Xts.shape[0])+"\n")
       filer.write('No of outliers in test set: '+str(dftsm2.shape[0])+"\n")
       filer.write('Q2F1/R2Pred: '+ str(r2pr)+"\n")
       filer.write('Q2F2: '+ str(r2pr2)+"\n")
       filer.write('rm2test: '+str(rm2ts)+"\n")
       filer.write('delta rm2test: '+str(drm2ts)+"\n")
       filer.write('RMSEP: '+str(RMSEP)+"\n")
       filer.write('MAE: '+str(dts)+"\n")
       filer.write('AARDts: '+str(aardts)+"\n")
       filer.write("\n")
    
       cfile.write('rm2LOO: '+str(rm2tr)+"\n")
       cfile.write('delta rm2LOO: '+str(drm2tr)+"\n")
       cfile.write('LCO_pred: '+str(lcorpred)+"\n")
       cfile.write('LCO_rm2: '+str(lcorm2)+"\n")
       cfile.write('LCO_drm2: '+str(lcodrm2)+"\n")
       cfile.write("\n")
       cfile.write('Test set results: '+"\n")
       cfile.write('Number of observations '+str(Xts.shape[0])+"\n")
       cfile.write('No of outliers in test set: '+str(dftsm2.shape[0])+"\n")
       cfile.write('Q2F1/R2Pred: '+ str(r2pr)+"\n")
       cfile.write('Q2F2: '+ str(r2pr2)+"\n")
       cfile.write('rm2test: '+str(rm2ts)+"\n")
       cfile.write('delta rm2test: '+str(drm2ts)+"\n")
       cfile.write('MAE: '+str(dts)+"\n")
       cfile.write('AARDts: '+str(aardts)+"\n")
       cfile.write('###########################Next################################'+"\n")
       cfile.write("\n")
    else:
        Xts=file2.iloc[:,1:]
        nts=file2.iloc[:,0:1]
        ytspr=pd.DataFrame(reg.predict(Xts[a]))
        ytspr.columns=['Pred']
        adts=apdom(Xts[a],Xtr[a])
        yadts=adts.fit()
        dfts=pd.concat([nts,Xts[a],ytspr,yadts],axis=1)
        #dfts.to_csv(str(c_)+"_sfslda_scpr.csv",index=False)

    return Xtr.shape[0],c,lcoq2,dloo,Xts.shape[0],r2pr,dts,mc
    

def trainsetfit2(X,y,filer,file1):
    cthreshold=thirdEntryTabThreer3c2.get()
    vthreshold=fourthEntryTabThreer5c2.get()
    max_steps=fifthBoxTabThreer6c2.get()
    #max_steps=int(X.shape[0]/5)
    flot=Criterion.get()
    forw=Criterion3.get()
    score=Criterion4.get()
    cvl=fifthBoxTabThreer7c2.get()
    #filer = open(str(c_)+"_results.txt","a")
    filer.write("Correlation cut-off "+str(cthreshold)+"\n")
    filer.write("Variance cut-off "+str(vthreshold)+"\n")
    filer.write("Maximum steps "+str(max_steps)+"\n")
    filer.write("Floating "+str(flot)+"\n")
    filer.write("Scoring "+str(score)+"\n")
    filer.write("Cross_validation "+str(cvl)+"\n")
    filer.write("% of CV increment "+str(fifthLabelTabThreer9c2.get())+"\n")
    lt=[0.0001]
    sqs=sq(X,y,float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))
    a1,b1=sqs.fit_()
    X[a1]=X[a1].reset_index().drop('index',axis=1)
    y=y.reset_index().drop('index',axis=1)
    file1=file1.reset_index().drop('index',axis=1)
       
    for i in range(1,len(a1)+1,1):
        sqs=sq(X[a1],y,float(cthreshold),float(vthreshold),i,flot,forw,score,int(cvl))
        a,b=sqs.fit_()
        cv=loo(X[a],y,file1)
        c,m,l=cv.cal()
        lt.append(c)
        val=(lt[len(lt)-1]-lt[len(lt)-2])
        #print(c)
        #print(val/lt[len(lt)-2]*100)
        val2=val/lt[len(lt)-2]*100   
        if val2<float(fifthLabelTabThreer9c2.get()):
           break
    if X[a].shape[1]>2:
        tb=X[a].corr()
        mx,mn=corr(tb)
    else:
        mx='Less number of descriptors'
        mn='Less number of descriptors'
    #tbn=str(a_)+'_corr.csv'
    #tb.to_csv(tbn)
    #pt.to_csv('pt_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    #dt.to_csv('dt.csv',index=False)
    #print(c,m,mx,mn)
    return a,b,c,m,mx,mn,l,filer
    #print(float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))

def enable3():
    N1B1_x['state']='normal'

firstLabelTabOne = tk.Label(tab1, text="Select Descriptor Data",font=("Helvetica", 12))
firstLabelTabOne.place(x=30,y=10)
firstEntryTabOne = tk.Entry(tab1,text='',width=50)
firstEntryTabOne.place(x=200,y=13)
b5=tk.Button(tab1,text='Browse', command=data1,font=("Helvetica", 10))
b5.place(x=520,y=10)

thirdLabelTabThreer2c2_1=Label(tab1, text='Mention number of fixed features',font=("Helvetica", 12))
thirdLabelTabThreer2c2_1.place(x=80,y=45)
thirdEntryTabThreer3c2_1=Entry(tab1)
thirdEntryTabThreer3c2_1.place(x=320,y=50)

thirdLabelTabThreer2c2_2=Label(tab1, text='Type the output folder name',font=("Helvetica", 12))
thirdLabelTabThreer2c2_2.place(x=80,y=75)
thirdEntryTabThreer3c2_2=Entry(tab1)
thirdEntryTabThreer3c2_2.place(x=320,y=80)

secondLabelTabOne_2=Label(tab1, text='Descriptor calculation method',font=('Helvetica 12 bold'))
secondLabelTabOne_2.place(x=220,y=100)

var1= IntVar()
C2 = Checkbutton(tab1, text = "Method-1",  variable=var1, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
#C2.grid(row=3, column=4)
C2.place(x=230,y=125)

var2= IntVar()
C2 = Checkbutton(tab1, text = "Method-2",  variable=var2, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
#C2.grid(row=3, column=4)
C2.place(x=320,y=125)

fourthLabelTabThreer4c2_1=Label(tab1, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2_1.place(x=200,y=155)
fourthEntryTabThreer5c2_1=Entry(tab1)
fourthEntryTabThreer5c2_1.place(x=315,y=155)

secondLabelTabOne_1=Label(tab1, text='Dataset division techniques',font=('Helvetica 12 bold'))
secondLabelTabOne_1.place(x=220,y=190)

Selection = StringVar()
Criterion_sel1 = ttk.Radiobutton(tab1, text='Point-out', variable=Selection, value='Point-out')
Criterion_sel2 = ttk.Radiobutton(tab1, text='Mixture-out', variable=Selection, value='Mixture-out')
Criterion_sel3 = ttk.Radiobutton(tab1, text='Compound-out', variable=Selection, value='Compound-out')
Criterion_sel1.place(x=140,y=215)
Criterion_sel2.place(x=270,y=215)
Criterion_sel3.place(x=420,y=215)

secondLabelTabOne=Label(tab1, text='Specify maximum seed value',font=("Helvetica", 12), justify='center')
secondLabelTabOne.place(x=150,y=240)
secondEntryTabOne=Entry(tab1)
secondEntryTabOne.place(x=360,y=243)


secondLabelTabOne_y=Label(tab1, text='Specify maximum interval',font=("Helvetica", 12), justify='center')
secondLabelTabOne_y.place(x=150,y=270)
secondEntryTabOne_y=Entry(tab1)
secondEntryTabOne_y.place(x=360,y=273)

#b6_x=tk.Button(tab1, text='Generate train-test sets', bg="orange", command=datasel,font=("Helvetica", 10))
#b6_x.place(x=310,y=300)


L1=Label(tab1, text='Stepwise multiple linear regression',font=("Helvetica 12 bold"))
L1.place(x=200,y=300)

thirdLabelTabThreer2c2=Label(tab1, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c2.place(x=220,y=330)
thirdEntryTabThreer3c2=Entry(tab1)
thirdEntryTabThreer3c2.place(x=345,y=330)

fourthLabelTabThreer4c2=Label(tab1, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2.place(x=220,y=355)
fourthEntryTabThreer5c2=Entry(tab1)
fourthEntryTabThreer5c2.place(x=345,y=355)

fifthLabelTabThreer6c2 = Label(tab1, text= 'Maximum steps',font=("Helvetica", 12))
fifthLabelTabThreer6c2.place(x=30,y=380)
fifthBoxTabThreer6c2= Spinbox(tab1, from_=0, to=100, width=5)
fifthBoxTabThreer6c2.place(x=150,y=380)

fifthLabelTabThreer8c2 = Label(tab1, text= '% of CV increment',font=("Helvetica", 12))
fifthLabelTabThreer8c2.place(x=200,y=380)
fifthLabelTabThreer9c2=Entry(tab1)
fifthLabelTabThreer9c2.place(x=345,y=380)

fifthLabelTabThreer7c2 = Label(tab1, text= 'Cross_validation',font=("Helvetica", 12))
fifthLabelTabThreer7c2.place(x=230,y=405)
fifthBoxTabThreer7c2= Spinbox(tab1, from_=0, to=100, width=5)
fifthBoxTabThreer7c2.place(x=355,y=405)

Criterion_Label = ttk.Label(tab1, text="Floating:",font=("Helvetica", 12))
Criterion = BooleanVar()
Criterion.set(False)
Criterion_Gini = ttk.Radiobutton(tab1, text='True', variable=Criterion, value=True)
Criterion_Entropy = ttk.Radiobutton(tab1, text='False', variable=Criterion, value=False)
Criterion_Label.place(x=230,y=430)
Criterion_Gini.place(x=300,y=430)
Criterion_Entropy.place(x=350,y=430)

Criterion_Label3 = ttk.Label(tab1, text="Forward:",font=("Helvetica", 12))
Criterion3 = BooleanVar()
Criterion3.set(True)
Criterion_Gini2 = ttk.Radiobutton(tab1, text='True', variable=Criterion3, value=True)
#Criterion_Gini2.pack(column=4, row=9, sticky=(W))
Criterion_Entropy2 = ttk.Radiobutton(tab1, text='False', variable=Criterion3, value=False)
Criterion_Label3.place(x=230,y=455)
Criterion_Gini2.place(x=300,y=455)
Criterion_Entropy2.place(x=350,y=455)


Criterion_Label4 = ttk.Label(tab1, text="Scoring:",font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion4 = StringVar()
Criterion4.set('r2')
Criterion_acc3 = ttk.Radiobutton(tab1, text='R2', variable=Criterion4, value='r2')
#Criterion_prec3 = ttk.Radiobutton(tab3, text='Precision', variable=Criterion4, value='precision')
Criterion_roc3 = ttk.Radiobutton(tab1, text='NMAE', variable=Criterion4, value='neg_mean_absolute_error')
Criterion_roc4 = ttk.Radiobutton(tab1, text='NMPD', variable=Criterion4, value='neg_mean_poisson_deviance')
Criterion_roc5 = ttk.Radiobutton(tab1, text='NMGD', variable=Criterion4, value='neg_mean_gamma_deviance')
Criterion_Label4.place(x=230,y=480)
Criterion_acc3.place(x=300,y=480)
Criterion_roc3.place(x=370,y=480)
Criterion_roc4.place(x=440,y=480)
Criterion_roc5.place(x=510,y=480)

b2=Button(tab1, text='Generate model', command=execute,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=300,y=505)


tab_parent.pack(expand=1, fill='both')

form.mainloop()