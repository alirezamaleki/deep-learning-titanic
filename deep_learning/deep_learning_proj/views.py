from django.shortcuts import render
from . import nn_predict

def home(request):
    return render(request,'index.html')

def results(request):
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    ticket = int(request.GET['ticket'])
    fare = int(request.GET['fare'])
    cabin = int(request.GET['cabin'])
    embarked = int(request.GET['embarked'])
    prediction = nn_predict.nn_prediction_model(sex, age, sibsp, parch, ticket, fare, cabin, embarked)
    return render(request, 'results.html', {"prediction": prediction})