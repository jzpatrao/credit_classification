import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from client_features import ClientFeatures
# import uvicorn

with open("trained_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Credit Default Prediction ",
        version="0.0.1",
        description="Classification with Logistic Regression. To try it out, click 'Try it out' under '/score' ",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/")
async def root():
    return {"message": "Hello World"}
    
@app.post('/score')
async def give_score(data:ClientFeatures):
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])
    score = model.predict_proba(df)[0][1]
    score = round(score, 2)
    if score < 0.4933:
        status = 'Accepted'
    else:
        status = 'Rejected'

    return {"risk_score": score, 
            'application_status': status}

# if __name__ == '__main__':
#     uvicorn.run(app)