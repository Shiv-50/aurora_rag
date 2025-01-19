from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")




TXT_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "Aurora'25.txt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# --- Load and Initialize Resources ---
print("Loading ...")
loader = TextLoader(TXT_FILE_PATH)  # Changed
documents = loader.load()

template = """Answer the following question based solely on the information provided in the context below. Do not use any outside knowledge. If the answer is not in the context, say 'I don't know.'
context:
{context}

Question:
{question}?
"""
# --- API Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Welcome to the PDF Chatbot API!"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question
    try:
        question = question

        prompt = template.format(context=documents[0].page_content, question=question)
        
        answer = model.generate_content(prompt)
        
        return QueryResponse(answer = answer.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
