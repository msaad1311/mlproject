from flask import Flask,render_template, request
from src.pipeline.testing_pipeline import DataCreation, PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data=DataCreation(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        df_for_prediction = data.get_dataframe()
        prediction_pipeline = PredictionPipeline()
        result = prediction_pipeline.prediction(df_for_prediction)
        print(result)
        return render_template('index.html',results = round(result[0]))
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)
    
        
