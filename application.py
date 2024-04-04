from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.pred_pipeline import CustomData, PredictPipline
from src.pipeline.training_pipeline import Training_pipeline

application = Flask(__name__)

app= application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods = ("GET", 'POST'))
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        )

        custom_df = data.get_data_as_dataframe()
        print(custom_df)
        
        predict_pipeline = PredictPipline()
        results = predict_pipeline.predict(custom_df)
        return render_template ('home.html', result = results[0])
    
@app.route('/upload', methods=['GET','POST'])
def upload_file():
    # Check if the POST request has the file part
    if request.method == 'GET':
        return render_template('train.html')
    else:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and file.filename.endswith('.csv'):
       
            df = pd.read_csv(file)

            train = Training_pipeline()
            r2score = train.Training(df)

            return render_template('train.html', result= r2score)
        else:
            return jsonify({'error': 'Invalid file format'})
        

    
if __name__ == "__main__":
    app.run(host="0.0.0.0")