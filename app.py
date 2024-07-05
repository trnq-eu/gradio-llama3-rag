import gradio as gr
import time
import random
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

llm = Groq(model='llama3-70b-8192', api_key=groq_api_key)


def chatbot(message, history):
    response = llm.complete(message)
    return response.text


gr.ChatInterface(
    chatbot,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title = 'Yes Man',
    description="Ask Yes Man any questions",
    theme='soft',
    examples=["Hello", "Am I cool", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn='Clear',
    multimodal=True
).launch()