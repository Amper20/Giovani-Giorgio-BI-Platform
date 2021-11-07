from flask import Flask
from flask import request
import Eval as evl
from dateutil.parser import parse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/future', methods=["POST"])
def getFuture():
    response = {}
    for dataset in evl.get_available_datasets():
        date = request.json["date"] 
        count = request.json["count"]
        response[dataset] = evl.get_predictions(parse(date).date(), count, dataset)
    
    return response 


@app.route('/suggestions', methods=["GET"])
def get_suggestions():    
    return {"data":evl.get_suggestions()}