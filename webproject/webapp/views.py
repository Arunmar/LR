from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# Create your views here.
def index(request):
    csv_filename = os.path.join(os.path.dirname(__file__), 'AI_model.csv')
    a = pd.read_csv(csv_filename)
    a = a[np.isfinite(a).all(1)]# just ignore
    x1 = a.iloc[:, 0].values.reshape(-1, 1)
    y = a.iloc[:, 2].values.reshape(-1, 1)
    lr = LinearRegression()
    model = lr.fit(x1, y)
    if request.method == "POST":
        us = request.POST.get('username')
        predict = model.predict([[us]])
        return render(request,'first.html',{"predict":predict[0][0],"score":model.score(x1,y)*100})
    return render(request, 'first.html')


def index2(request):
    csv_filename = os.path.join(os.path.dirname(__file__), 'AI_model.csv')
    a = pd.read_csv(csv_filename)
    x = a.iloc[:, :-1].values
    y = a.iloc[:, 2].values
    lr = LinearRegression()
    model = lr.fit(x, y)
    if request.method == "POST":
        us = request.POST.get('username')
        us2 = request.POST.get('username2')
        predict = model.predict([[us,us2]])
        return render(request,'second.html',{"predict":predict,"score":model.score(x,y)*100})
    return render(request, 'second.html')


def index3(request):
    csv_filename = os.path.join(os.path.dirname(__file__), 'AI_model2.csv')
    a = pd.read_csv(csv_filename)
    x = a.iloc[:, :-1].values
    y = a.iloc[:, 2].values
    lr = LinearRegression()
    model = lr.fit(x, y)
    if request.method == "POST":
        us = request.POST.get('username')
        us2 = request.POST.get('username2')
        predict = model.predict([[us,us2]])
        return render(request,'third.html',{"predict":predict,"score":model.score(x,y)*100})
    return render(request, 'third.html')
# for 1
# a = pd.read_csv('AI_model2.csv')
# x1 = a.iloc[:, 0].values.reshape(-1,1)
# y = a.iloc[:, 2].values.reshape(-1,1)
# lr = LinearRegression()
# model = lr.fit(x1,y)
# predict = model.predict([[900]])
# print(x1)
# print(y)
# print(predict)
# print(model.score(x1,y))# R square coefficient


#for 2
# x = a.iloc[:, :-1].values
# y = a.iloc[:, 2].values
# print(x)
# print(y)
# lr = LinearRegression()
# model = lr.fit(x,y)
# predict = model.predict([[1300  ,  40]])
# print(predict)
# print(model.score(x,y))
# import seaborn as sb
# sb.pairplot(a)
# plt.show()



# # for 1 input
# x1 = a.iloc[:,:0].values.reshape(-1,1)
# # print(x1)
# #x1 = a.iloc[:, :1].values.reshape(-1,1)
# y = a.iloc[:,6].values.reshape(-1,1)
# print(y)
# lr = LinearRegression()
# model = lr.fit(x1,y)
# plt.scatter(x1,y)
# plt.plot(x1,model.predict(x1),color="red")
# plt.xlabel("FSP tool rotation speed (rpm)")
# plt.ylabel("Surface Hardness")
# plt.show()
# a = model.predict([[900]])
#print(a)