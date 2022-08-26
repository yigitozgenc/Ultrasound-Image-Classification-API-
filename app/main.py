from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from app.utils.classification import classify_image
from fastapi.middleware.cors import CORSMiddleware

app_desc = """<h2>This API created for PONSTech hiring Interview`</h2>
<h2>Ultrasound Image Classification with EmergencyNet Architecture</h2>
<br>by Mehmet Yiğit Özgenç"""

app= FastAPI(title='Ultrasound Data', description=app_desc)

origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify")
async def predict(file: UploadFile = File(...)):
    classification_results = classify_image(await file.read())
    return classification_results


if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8001)    
