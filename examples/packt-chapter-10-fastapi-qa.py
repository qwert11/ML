# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH10/CH10a_fastAPI_Transformer_model_serving/main.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

class QADataModel(BaseModel):
    question: str
    context: str

app = FastAPI()
model_name = "distilbert-base-cased-distilled-squad"
model = pipeline(
    model=model_name,
    tokenizer=model_name,
    task="question-answering",
)

@app.post("/question_answering")
async def qa(input_data: QADataModel):
    result = model(
        question=input_data.question,
        context=input_data.context,
    )
    return {"answer": result["answer"]}

if __name__ == "__main__":
    uvicorn.run("main:app", workers=1)
