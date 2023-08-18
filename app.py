import os
import pandas as pd
from wsgiref import simple_server
from flask import Flask, render_template, request
from src.feature_extraction import generate_features
from src.predict import predict

HOST = "0.0.0.0"
PORT = 5000

app = Flask(__name__)

def _predict(question1, question2):
    df = pd.DataFrame(columns=['id', 'qid1', 'qid2', 'question1', 'question2'])
    df.loc[0, :] = [0, 1, 2, question1, question2] 
    try:
        df = generate_features(df, is_train=False)
        prediction = predict(df.drop(columns=['id']), proba=True)[:, 1]
        print(prediction)
    except Exception as e:
        print(e)
        return "Sorry, something went wrong."
    
    if prediction > 0.75:
        result = "Similar"
    elif prediction > 0.5:
        result = f"Similar, only {round(prediction[0]*100, 1)}% confident"
    elif prediction > 0.25:
        result = f"Not similar, only {round(1-prediction[0], 1)*100}% confident"
    else:
        result = "Not similar"
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def get_predictions():
    if 'POST' == request.method:
        question1 = request.form['question1']
        question2 = request.form['question2']
        return _predict(question1, question2)
    
    return None
    
    
if "__main__" == __name__:
    port = int(os.getenv("PORT", PORT))
    server = simple_server.make_server(HOST, port=port, app=app)
    server.serve_forever()
