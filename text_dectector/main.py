import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

# from text_dectector.predict import TextDetectorAPI
#
# app = FastAPI()
#
# predictor = TextDetectorAPI(use_gpu=False)
#
#
# class BoxResult(BaseModel):
#     box: list
#
#
# @app.post("/predict", response_model=BoxResult)
# async def predict(file: UploadFile = File(...)):
#     try:
#         image = await file.read()
#
#         if file.content_type != "application/pdf":
#             image = Image.open(BytesIO(image)).convert("RGB")
#             image = np.array(image)
#         result = predictor.predict(image)
#         result = result[0]
#         return {"box": result}
#     except Exception as e:
#         return {"error": str(e)}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
