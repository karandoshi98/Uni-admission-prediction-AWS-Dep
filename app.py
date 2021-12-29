
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            gre_score=float(request.form['gre_score'])
            toefl_score = float(request.form['toefl_score'])
            university_rating = float(request.form['university_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            is_research = request.form['research']
            if(is_research=='yes'):
                research=1
            else:
                research=0
            filename = 'admission_pred_Elastic_LR_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file

            df = pd.read_csv('Admission_Prediction.csv')
            # print("DF loaded")
            df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())
            df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
            df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mean())
            # print("NA values filled with their respective mean")
            df.drop('Serial No.', axis=1, inplace=True)
            # print("Column : serial no dropped")
            y = df['Chance of Admit']
            df.drop(columns=['Chance of Admit'], inplace=True)
            x = df
            sc = StandardScaler()
            x_sc = sc.fit_transform(x)
            prediction=loaded_model.predict(sc.transform([[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]]))
            # print("Test" + str(loaded_model.predict(sc.transform([[337.000000, 118.0, 4.0, 4.5, 4.5, 9.65, 1]]))))

            # print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=round(100*prediction[0]))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app