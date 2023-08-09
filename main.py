from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app=FastAPI()
#settings-----------------------------------------
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


#models--------------------------------------------
class model_input(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int

#loading the model--------------------------------------
ml_model=pickle.load(open('diabetes_model.sav','rb'))


#views---------------------------------------------------
@app.post('/diabetes_prediction')
def predict(input_params: model_input):
    input_list = [
        input_params.Pregnancies,
        input_params.Glucose,
        input_params.BloodPressure,
        input_params.SkinThickness,
        input_params.Insulin,
        input_params.BMI,
        input_params.DiabetesPedigreeFunction,
        input_params.Age
    ]

    prediction = ml_model.predict([input_list])
    if prediction == 0:
        return 'The Person is not Diabetic'
    else:
        return 'The Person is Diabetic'