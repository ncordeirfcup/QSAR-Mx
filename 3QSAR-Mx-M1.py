import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import os
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
#from kmca import kmca
from sequential_selection import stepwise_selection as sq
from loo import loo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from applicability import apdom
import math
import numpy as np
from matplotlib import pyplot
import threading


form = tk.Tk()
form.title("QSAR-mx")
form.geometry("650x350")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="Module 1:Data preparation")
tab_parent.add(tab2, text="Module 1:Model development")

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


def datatr():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    global col1
    col1 = list(file1.head(0))
    
def datats():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    #global col2
    #col2 = list(file2.head(0))


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

def shuffling(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

    
def method1(x1,x2,f1,f2):
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

def variance(X,threshold):
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X

def check_data(n):
    ds=file.iloc[:,0:n+4]
    filex1=file.drop_duplicates(subset=list(ds.columns)[1::])
    fdupl=file.drop(filex1.index.values)      
    if fdupl.shape[0]>0:
       messagebox.showinfo('Warning','Duplicate samples found in the data, will be removed')
       fdupl.to_csv(str(a_)+'duplicates.csv', index=False)
       filex1.to_csv(str(a_)+'duplicates_removed.csv', index=False)
    else:
       filex1=file
    return filex1
    

def pointout(df,n,c1,c2,sp):
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

def mixtureout(df,n,c1,c2,sp):
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

def compout(df,n,c1,c2,sp):
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
        messagebox.showinfo('Error','Select other values')
    else:
        fdftsa=pd.concat([dftsa,dftsa2],axis=0)
        fdftra=df.drop(fdftsa.index.values)
    return fdftra,fdftsa

def wait_end(label, tk_var_end, num=0):
    label["text"] = "Processing " + " ." * num
    num += 1
    if num == 4:
        num = 0
    if not tk_var_end.get():
        form.after(500, wait_end, label, tk_var_end, num)


def execute():
    tk_process_lbl = tk.Label(form,font=('Helvetica 12 bold'),fg="blue")
    tk_process_lbl.pack()
    tk_process_lbl.place(x=470,y=305)

    tk_var_end = tk.BooleanVar(False)
    wait_end(tk_process_lbl, tk_var_end)
    process = threading.Thread(
        target=writefilex,
        kwargs=(dict(callback=lambda: tk_var_end.set(True)))
    )
    process.start()

    form.wait_variable(tk_var_end)
    form.after(500, tk_process_lbl.config, dict(text='Process completed'))
    #form.after(1500, form.quit)
        

def execute2():
    tk_process_lbl = tk.Label(form,font=('Helvetica 12 bold'),fg="red")
    tk_process_lbl.pack()
    tk_process_lbl.place(x=470,y=305)

    tk_var_end = tk.BooleanVar(False)
    wait_end(tk_process_lbl, tk_var_end)
    process = threading.Thread(
        target=datasel,
        kwargs=(dict(callback=lambda: tk_var_end.set(True)))
    )
    process.start()

    form.wait_variable(tk_var_end)
    form.after(500, tk_process_lbl.config, dict(text='Process completed'))
    #form.after(1500, form.quit)

def datasel(callback):
    n=int(thirdEntryTabThreer3c2_1.get())
    filex1=check_data(n)
    sn=filex1.iloc[:,0:1]
    inc=filex1.iloc[:,0:n+5]
    inc2=filex1.iloc[:,1:n+4]
    inc3=file.iloc[:,1:5]
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
        trp,tsp=pointout(fileam,int(secondEntryTabOne.get()),c1,c2,int(secondEntryTabOne_y.get()))
        method='po'
    elif Selection.get()=='Mixture-out':
        trp,tsp=mixtureout(fileam,int(secondEntryTabOne.get()),c1,c2,int(secondEntryTabOne_y.get()))
        method='mo'
    elif Selection.get()=='Compound-out':
        trp,tsp=compout(fileam,int(secondEntryTabOne.get()),c1,c2,int(secondEntryTabOne_y.get()))
        method='co'
    trdp=trp.drop(inc3.columns,axis=1)
    tsdp=tsp.drop(inc3.columns,axis=1)
    ###
    sc=trdp.iloc[:,1:n]
    trdp=trdp.drop(sc,axis=1)
    trdp=pd.concat([trdp,sc],axis=1)
    ###
    sc=tsdp.iloc[:,1:n]
    tsdp=tsdp.drop(sc,axis=1)
    tsdp=pd.concat([tsdp,sc],axis=1)
    ###
    trsp=trp[inc.iloc[:,0:5].columns]
    trsp['Set']='Train'
    tssp=tsp[inc.iloc[:,0:5].columns]
    tssp['Set']='Test'
    tsp=pd.concat([trsp,tssp],axis=0)
    tsp.to_csv(str(a_)+str('_')+str(method)+'_division.csv',index=False)
    savename1= str(a_)+str(method)+'_tr.csv'
    trdp.to_csv(savename1,index=False)
    #savename2 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save testset file")
    savename2= str(a_)+str(method)+'_ts.csv'
    tsdp.to_csv(savename2,index=False)
    callback()
  

def trainsetfit2(X,y):
    cthreshold=thirdEntryTabThreer3c2.get()
    vthreshold=fourthEntryTabThreer5c2.get()
    max_steps=fifthBoxTabThreer6c2.get()
    flot=Criterion.get()
    forw=Criterion3.get()
    score=Criterion4.get()
    cvl=fifthBoxTabThreer7c2.get()
    filer = open(str(c_)+"_results.txt","w")
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
    #print(c)
    tb=X[a].corr()
    mx,mn=corr(tb)
    tbn=str(c_)+'_corr.csv'
    tb.to_csv(tbn)
    #pt.to_csv('pt_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    #dt.to_csv('dt.csv',index=False)
    return a,b,c,m,mx,mn,l,filer
    #print(float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))


def writefilex(callback):
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    a,b,c,m,mx,mn,l,filer=trainsetfit2(Xtr,ytr)
    reg.fit(Xtr[a],ytr)
    r2=reg.score(Xtr[a],ytr)
    ypr=pd.DataFrame(reg.predict(Xtr[a]))
    ypr.columns=['Pred']
    rm2tr,drm2tr=rm2(ytr,l).fit()

    trdf=pd.concat([ytr,pd.DataFrame(ypr)],axis=1)
    trdf.columns=['Active','Predict']
    trdf['diff']=abs(trdf['Active']-trdf['Predict'])
    aard=(100*(trdf['diff']/trdf['Active'])).sum()/(ytr.shape[0])
    #aard=(100*(abs((ytr-ypr)/ytr))).sum()/(ytr.shape[0])
    #savefile.to_csv('savefile.csv',index=False)
    d=mean_absolute_error(ytr,ypr)
    e=(mean_squared_error(ytr,ypr))**0.5
    adstr=apdom(Xtr[a],Xtr[a])
    yadstr=adstr.fit() 
    df=pd.concat([ntr,Xtr[a],ytr,ypr,l,yadstr],axis=1)
    df.to_csv(str(c_)+"_sfslda_trpr.csv",index=False)
    
    #filer = open(str(c_)+"_sfslda.txt","w")
    
    filer.write("Sub-training set results "+"\n")
    filer.write("\n")
    filer.write("Selected features are:"+str(a)+"\n")
    filer.write("Statistics:"+str(b)+"\n")
    filer.write('Training set results: '+"\n")
    filer.write('Maxmimum intercorrelation between descriptors: '+str(mx)+"\n")
    filer.write('Minimum intercorrelation between descriptors: '+str(mn)+"\n")
    filer.write('MAE: '+str(d)+"\n")
    filer.write('RMSE: '+str(e)+"\n")
    filer.write('Q2LOO: '+str(c)+"\n")
    filer.write('AARD: '+str(aard)+"\n")
    
    if ytr.columns[0] in file2.columns:
       Xts=file2.iloc[:,2:]
       yts=file2.iloc[:,1:2]
       nts=file2.iloc[:,0:1]
       ytspr=pd.DataFrame(reg.predict(Xts[a]))
       ytspr.columns=['Pred']
       rm2ts,drm2ts=rm2(yts,ytspr).fit()
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
       dfts=pd.concat([nts,Xts[a],yts,ytspr,yadts],axis=1)
       dfts.to_csv(str(c_)+"_sfslda_tspr.csv",index=False)
       filer.write('rm2LOO: '+str(rm2tr)+"\n")
       filer.write('delta rm2LOO: '+str(drm2tr)+"\n")
       filer.write("\n")
       filer.write('Test set results: '+"\n")
       filer.write('Number of observations '+str(Xts.shape[0])+"\n")
       filer.write('Q2F1/R2Pred: '+ str(r2pr)+"\n")
       filer.write('Q2F2: '+ str(r2pr2)+"\n")
       filer.write('rm2test: '+str(rm2ts)+"\n")
       filer.write('delta rm2test: '+str(drm2ts)+"\n")
       filer.write('RMSEP: '+str(RMSEP)+"\n")
       filer.write('AARDts: '+str(aardts)+"\n")
       filer.write("\n")
       pyplot.figure(figsize=(15,10))
       pyplot.scatter(ytr,ypr, label='Train', color='blue')
       pyplot.plot([ytr.min(), ytr.max()], [ytr.min(), ytr.max()], 'k--', lw=4)
       pyplot.scatter(yts,ytspr, label='Test', color='red')
       pyplot.ylabel('Predicted values',fontsize=28)
       pyplot.xlabel('Observed values',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       rocn=str(c_)+'_scatter.png'
       pyplot.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
    else:
        Xts=file2.iloc[:,1:]
        nts=file2.iloc[:,0:1]
        ytspr=pd.DataFrame(reg.predict(Xts[a]))
        ytspr.columns=['Pred']
        adts=apdom(Xts[a],Xtr[a])
        yadts=adts.fit()
        dfts=pd.concat([nts,Xts[a],ytspr,yadts],axis=1)
        dfts.to_csv(str(c_)+"_sfslda_scpr.csv",index=False)
    if var3.get():
        ls=[]
        nr=int(N1B1_x.get())
        for i in range(0,nr):
            yr=shuffling(ytr)
            reg.fit(Xtr[a],yr)
            ls.append(reg.score(Xtr[a],yr))
        rr=np.mean(ls)
        reg.score(Xtr[a],ytr)
        #r2=b.rsquared
        crp2= math.sqrt(r2)*math.sqrt(r2-rr)
        filer.write('Crp2 after '+str(nr) + ' run: '+str(crp2)+"\n")
    callback()

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

secondLabelTabOne_2=Label(tab1, text='Descriptor calculation method',font=('Helvetica 12 bold'))
secondLabelTabOne_2.place(x=220,y=80)

var1= IntVar()
C2 = Checkbutton(tab1, text = "Method-1",  variable=var1, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
#C2.grid(row=3, column=4)
C2.place(x=230,y=105)

var2= IntVar()
C2 = Checkbutton(tab1, text = "Method-2",  variable=var2, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
#C2.grid(row=3, column=4)
C2.place(x=320,y=105)

fourthLabelTabThreer4c2_1=Label(tab1, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2_1.place(x=200,y=135)
fourthEntryTabThreer5c2_1=Entry(tab1)
fourthEntryTabThreer5c2_1.place(x=315,y=135)

secondLabelTabOne_1=Label(tab1, text='Dataset division techniques',font=('Helvetica 12 bold'))
secondLabelTabOne_1.place(x=220,y=170)

Selection = StringVar()
Criterion_sel1 = ttk.Radiobutton(tab1, text='Point-out', variable=Selection, value='Point-out')
Criterion_sel2 = ttk.Radiobutton(tab1, text='Mixture-out', variable=Selection, value='Mixture-out')
Criterion_sel3 = ttk.Radiobutton(tab1, text='Compound-out', variable=Selection, value='Compound-out')
Criterion_sel1.place(x=140,y=195)
Criterion_sel2.place(x=270,y=195)
Criterion_sel3.place(x=420,y=195)

secondLabelTabOne=Label(tab1, text='Specify interval',font=("Helvetica", 12), justify='center')
secondLabelTabOne.place(x=180,y=250)
secondEntryTabOne=Entry(tab1)
secondEntryTabOne.place(x=320,y=253)


secondLabelTabOne_y=Label(tab1, text='Specify seed value',font=("Helvetica", 12), justify='center')
secondLabelTabOne_y.place(x=150,y=220)
secondEntryTabOne_y=Entry(tab1)
secondEntryTabOne_y.place(x=320,y=223)

b6_x=tk.Button(tab1, text='Generate train-test sets', bg="orange", command=execute2,font=("Helvetica", 10))
b6_x.place(x=310,y=280)


####TAB2##########
firstLabelTabThree = tk.Label(tab2, text="Select training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=95,y=10)
firstEntryTabThree = tk.Entry(tab2, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab2,text='Browse', command=datatr,font=("Helvetica", 10))
b3.place(x=480,y=10)

secondLabelTabThree = tk.Label(tab2, text="Select test/screening set",font=("Helvetica", 12))
secondLabelTabThree.place(x=45,y=40)
secondEntryTabThree = tk.Entry(tab2,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab2,text='Browse', command=datats,font=("Helvetica", 10))
b4.place(x=480,y=40)

L1=Label(tab2, text='Stepwise multiple linear regression',font=("Helvetica 12 bold"))
L1.place(x=200,y=80)

thirdLabelTabThreer2c2=Label(tab2, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c2.place(x=220,y=110)
thirdEntryTabThreer3c2=Entry(tab2)
thirdEntryTabThreer3c2.place(x=345,y=110)

fourthLabelTabThreer4c2=Label(tab2, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2.place(x=220,y=135)
fourthEntryTabThreer5c2=Entry(tab2)
fourthEntryTabThreer5c2.place(x=345,y=135)

fifthLabelTabThreer6c2 = Label(tab2, text= 'Maximum steps',font=("Helvetica", 12))
fifthLabelTabThreer6c2.place(x=30,y=160)
fifthBoxTabThreer6c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer6c2.place(x=150,y=160)

fifthLabelTabThreer8c2 = Label(tab2, text= '% of CV increment',font=("Helvetica", 12))
fifthLabelTabThreer8c2.place(x=200,y=160)
fifthLabelTabThreer9c2=Entry(tab2)
fifthLabelTabThreer9c2.place(x=345,y=160)

var3= IntVar()
N1 = Checkbutton(tab2, text = "Y-randomization",  variable=var3, \
                 font=("Helvetica", 12), command=enable3)
N1.place(x=480, y=110)

N1B1 = Label(tab2, text= 'Number of Runs',font=("Helvetica", 12))
N1B1.place(x=490,y=130)
N1B1_x=Entry(tab2,state=DISABLED)
N1B1_x.place(x=490,y=160)

fifthLabelTabThreer7c2 = Label(tab2, text= 'Cross_validation',font=("Helvetica", 12))
fifthLabelTabThreer7c2.place(x=230,y=185)
fifthBoxTabThreer7c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer7c2.place(x=355,y=185)

Criterion_Label = ttk.Label(tab2, text="Floating:",font=("Helvetica", 12))
Criterion = BooleanVar()
Criterion.set(False)
Criterion_Gini = ttk.Radiobutton(tab2, text='True', variable=Criterion, value=True)
Criterion_Entropy = ttk.Radiobutton(tab2, text='False', variable=Criterion, value=False)
Criterion_Label.place(x=230,y=210)
Criterion_Gini.place(x=300,y=210)
Criterion_Entropy.place(x=350,y=210)

Criterion_Label3 = ttk.Label(tab2, text="Forward:",font=("Helvetica", 12))
Criterion3 = BooleanVar()
Criterion3.set(True)
Criterion_Gini2 = ttk.Radiobutton(tab2, text='True', variable=Criterion3, value=True)
#Criterion_Gini2.pack(column=4, row=9, sticky=(W))
Criterion_Entropy2 = ttk.Radiobutton(tab2, text='False', variable=Criterion3, value=False)
Criterion_Label3.place(x=230,y=235)
Criterion_Gini2.place(x=300,y=235)
Criterion_Entropy2.place(x=350,y=235)


Criterion_Label4 = ttk.Label(tab2, text="Scoring:",font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion4 = StringVar()
Criterion4.set('r2')
Criterion_acc3 = ttk.Radiobutton(tab2, text='R2', variable=Criterion4, value='r2')
#Criterion_prec3 = ttk.Radiobutton(tab3, text='Precision', variable=Criterion4, value='precision')
Criterion_roc3 = ttk.Radiobutton(tab2, text='NMAE', variable=Criterion4, value='neg_mean_absolute_error')
Criterion_roc4 = ttk.Radiobutton(tab2, text='NMPD', variable=Criterion4, value='neg_mean_poisson_deviance')
Criterion_roc5 = ttk.Radiobutton(tab2, text='NMGD', variable=Criterion4, value='neg_mean_gamma_deviance')
Criterion_Label4.place(x=230,y=260)
Criterion_acc3.place(x=300,y=260)
Criterion_roc3.place(x=370,y=260)
Criterion_roc4.place(x=440,y=260)
Criterion_roc5.place(x=510,y=260)

b2=Button(tab2, text='Generate model', command=execute,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=350,y=285)


tab_parent.pack(expand=1, fill='both')

form.mainloop()