import gradio as gr
import time
import random
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.nomic import NomicEmbedding
from dotenv import load_dotenv
import os
import openai


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
nomic_api_key = os.getenv('NOMIC_KEY')
openai.api_key=os.getenv('OPEN_AI')

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=256,
    model_name="nomic-embed-text-v1.5",
)

# Update the global settings
Settings.embed_model = embed_model

llm = Groq(model='llama3-70b-8192', api_key=groq_api_key)

Settings.llm = llm

chroma_client =chromadb.Client()
# chroma_client.delete_collection('uploaded-docs')
chroma_collection = chroma_client.create_collection('uploaded-docs')



def process_documents(files):
    documents = SimpleDirectoryReader(input_files=files).load_data()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        embed_model=embed_model)
    return index

def chatbot(message, history, index):
    if index is None:
        return "Carica i tuoi documenti."
    query_engine = index.as_query_engine()
    # response = llm.complete(message)
    response = query_engine.query(message)
    return str(response)

with gr.Blocks() as demo:
    index = gr.State(None)
    gr.Markdown("# Parla con i tuoi documenti")
    with gr.Row():
        file_output = gr.File(file_count="multiple", label="Carica i documenti")
        upload_button = gr.Button("Processa i documenti")
        
    chatbot_interface = gr.Chatbot(height=300)
    msg = gr.Textbox(placeholder="Fammi delle domande sui tuoi documenti. Non ti deluderò!")
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, index):
        bot_message = chatbot(history[-1][0], history, index)
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot_interface], [msg, chatbot_interface], queue=False).then(
        bot, [chatbot_interface, index], chatbot_interface
    )
    clear.click(lambda: None, None, chatbot_interface, queue=False)

    def process_and_update(files):
        if files:
            new_index = process_documents([file.name for file in files])
            return new_index
        return None
    
    upload_button.click(process_and_update, inputs=[file_output], outputs=[index])

demo.launch()
# print(Settings.embed_model)
# print(Settings.llm)

