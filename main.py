# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sklearn as sklearn
from sklearn.preprocessing import LabelEncoder

def readCSV (StringName) :
    f = pd.read_csv(StringName)
    return f
def IDDropper (dataFrame):
    dataFrame.drop('Loan_ID',axis=1,inplace=True)
    return dataFrame

def Concater (dataFrame1, dataFrame2) :
    X = pd.concat([dataFrame1,dataFrame2], axis=1)
    return X


def YTransformToBin(dataFrame):
    output_vals = {'Y':1, 'N':0}
    output = dataFrame['Loan_Status']
    dataFrame.drop('Loan_Status',axis=1, inplace=True)
    output=output.map(output_vals)
    return output

def List_CatTransformer(dataFrame):
    le = LabelEncoder()
    for col in dataFrame :
        dataFrame[col]= le.fit_transform(dataFrame[col])
    return dataFrame

def list_splitter (file):
    ListeCategorique =[]
    ListeNumerique = []

    for col,ty_pe in enumerate(file.dtypes):
        if ty_pe==object :
            ListeCategorique.append(file.iloc[:,col])
        else :
            ListeNumerique.append(file.iloc[:,col])

    return (ListeNumerique,ListeCategorique)

def ListToDataFrame(List):
    List = pd.DataFrame(List).transpose()
    return List

def FillNullValuesForCat (dataFrame):
    dataFrame=dataFrame.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return dataFrame

def FillNullValuesForNum (dataFrame):
    dataFrame.fillna(method="bfill",inplace=True)
    return dataFrame


file = readCSV("train_u6lujuX_CVtuZ9i.csv")
pd.set_option('display.max_rows',file.shape[0]+1)
print (file)
pd.set_option('display.max_rows',10)
print (file)


#file.info()
#print(file.isnull().sum().sort_values(ascending=False))

#print(file.describe())

#print(file.describe(include='0'))


l,l2 = list_splitter(file)
l=ListToDataFrame(l)
l2=ListToDataFrame(l2)

l=FillNullValuesForNum(l)
l2=FillNullValuesForCat(l2)
print (l)
print("---------------------")
print(l2)
#print(l2.isnull().sum().sort_values(ascending=False))
#print(l.isnull().sum().sort_values(ascending=False))

outs = YTransformToBin(l2)
print(outs)
print (l2)
l2=List_CatTransformer(l2)
print(l2)
l2=IDDropper(l2)
print(l2)

X= Concater(l,l2)
print()