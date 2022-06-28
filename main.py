from flask import Flask, request
import pickle
from google.cloud import storage
import numpy as np

storage_client = storage.Client()
bucket = storage_client.get_bucket('model-bucket-iris-yaniela')
blob = bucket.blob('flowers-pkl')
blob.download_to_filename('/tmp/flowers-pkl')
model_pk = pickle.load(open('/tmp/flowers-pkl', 'rb'))


app=Flask(__name__)

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