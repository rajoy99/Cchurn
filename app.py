import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('mamluk.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predictioned = model.predict(final_features)

    if predictioned[0]==1:
        result="This Customer is likely to go away(a probable churn)"
    else:
        result="This Customer will stay"


    return render_template("result.html", prediction = result)

     

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)




if __name__ == "__main__":
    app.run(debug=True)