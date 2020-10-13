import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import os
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np

form = tk.Tk()
form.title("MOOP")
form.geometry("600x180")

tab_parent = ttk.Notebook(form)
tab1 = ttk.Frame(tab_parent)
tab_parent.add(tab1, text="Multi-objective optimisation")
initialdir=os.getcwd()

def dataob():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select observed properties")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    #global col1
    #col1 = list(file1.head(0))
    
def datapr():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select predicted properties")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    #global col2
    #col2 = list(file2.head(0))

def calu(df,prop):
    ##undesired values based on observed properties
    umn=file1[file1.iloc[:,2:3].columns[0]].max()  ##3.015
    umx=file1[file1.iloc[:,2:3].columns[0]].min()  ##0.763
    umed=file1[file1.iloc[:,2:3].columns[0]].median()  ##1.806
    udn=(umed-umn)/(umx-umn)
    un1=np.log(udn)
    us=np.log(0.5)/un1
    ##undesired values based on predicted properties
    ln=[]
    for i in df[prop]:
        if i>=umx:
           ln.append(((i-umn)/(umx-umn))**us) ##IF(S2<=0.763,1,((S2-3.015)/(0.763-3.015))^1.114)
        else:
           ln.append(1)
    col=pd.DataFrame(ln)
    col.columns=['Des_u']
    nt=pd.concat([df.iloc[:,0:1],col],axis=1)
    nt=nt.fillna(0)
    nt1=nt.sort_values(df.iloc[:,0:1].columns[0],ascending=True)
    return nt1

def cald(df,prop):
    ##desired values
    dmn=file1[file1.iloc[:,1:2].columns[0]].min()
    dmx=file1[file1.iloc[:,1:2].columns[0]].max()
    dmed=file1[file1.iloc[:,1:2].columns[0]].median()
    dn=(dmed-dmn)/(dmx-dmn)
    dn1=np.log(dn)
    ds=np.log(0.5)/dn1
    ln=[]
    for i in df[prop]:
        if i>dmx:
           ln.append(1)
        elif i>=dmn and i<dmx:
           ln.append(((i-dmn)/(dmx-dmn))**ds)
        else:
           ln.append(0)
    col=pd.DataFrame(ln)
    col.columns=['Des_d']
    nt=pd.concat([df,col],axis=1)
    nt=nt.fillna(0)
    nt1=nt.sort_values(df.iloc[:,0:1].columns[0],ascending=True)
    return nt1

def execute():
    md=pd.merge(cald(file2,file2.iloc[:,1:2].columns[0]),calu(file2,file2.iloc[:,2:3].columns[0]),on=file2.iloc[:,0:1].columns[0],how='left')
    md['Global_des']=(md['Des_u']*md['Des_d'])**0.5
    fname=str(thirdEntryTabThree.get())
    md.to_csv(fname+'.csv',index=False)
    
    
firstLabelTabThree = tk.Label(tab1, text="Select observed properties",font=("Helvetica", 12))
firstLabelTabThree.place(x=35,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=240,y=13)
b3=tk.Button(tab1,text='Browse', command=dataob,font=("Helvetica", 10))
b3.place(x=490,y=10)

secondLabelTabThree = tk.Label(tab1, text="Select predicted properties*",font=("Helvetica", 12))
secondLabelTabThree.place(x=35,y=40)
secondEntryTabThree = tk.Entry(tab1,width=40)
secondEntryTabThree.place(x=240,y=43)
b4=tk.Button(tab1,text='Browse', command=datapr,font=("Helvetica", 10))
b4.place(x=490,y=40)

thirdLabelTabThree = tk.Label(tab1, text="Mention output result file name",font=("Helvetica", 12))
thirdLabelTabThree.place(x=95,y=70)
thirdEntryTabThree = tk.Entry(tab1,width=20)
thirdEntryTabThree.place(x=320,y=73)

forthLabelTabThree = tk.Label(tab1, text="*Note: The predicted properties must be obtained from models developed with the observed properties",font=("Helvetica 9 italic"))
forthLabelTabThree.place(x=10,y=130)

b2=Button(tab1, text='Submit', command=execute,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=340,y=100)

tab_parent.pack(expand=1, fill='both')

form.mainloop()