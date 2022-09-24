from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__) # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            df = pd.read_csv('Admission_Prediction.csv')
            df.drop(columns=['Serial No.', 'Chance of Admit'], axis=1, inplace=True)
            scaler = StandardScaler()
            scaler.fit_transform(df)
            gre_score = float(request.form['gre_score'])
            toefl_score = float(request.form['toefl_score'])
            university_rating = float(request.form['university_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            is_research = request.form['research']
            if (is_research=='yes'):
                is_research = 1
            else:
                is_research = 0
            file = 'multi_linear.pickle'
            model = pickle.load(open(file, 'rb'))
            inputs = scaler.transform([[gre_score, toefl_score, university_rating, sop, lor, cgpa, is_research]])
            prediction = model.predict(inputs)
            #print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=round(100*prediction[0]))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__=="__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app
