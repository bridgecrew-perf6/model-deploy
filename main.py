from flask import Flask, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression



model_pk = pickle.load(open('flowers-pkl', 'rb'))

app=Flask(__name__)


@app.route("/hello_world")
def hello():
    return "hello world"

@app.route("/add", methods=["GET"])
def add_GET():
    a=request.args.get('a')
    b=request.args.get('b')
    return str(int(a)+int(b))

@app.route("/addPost", methods=["POST"])
def add_POST():
    data=request.get_json()
    a=data['a']
    b=data['b']
    return str(int(a)+int(b))


#http:localhost:5000/api_predict
@app.route("/api_predict", methods=["POST","GET"])
def api_predict():
    if request.method=="POST":
       data=request.get_json()
       sepal_length=data['sepal.length']
       sepal_width=data['sepal.width']
       petal_length=data['petal.length']
       petal_width=data['petal.width']
       input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
       prediction=model_pk.predict(input_data) 
       return str(prediction)
    else: return "Please send a POST request"   
    
app.run()