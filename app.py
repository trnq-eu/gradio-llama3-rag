import gradio as gr
import time
import random
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.llms.groq import Groq
# from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os
# import openai


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
# openai.api_key=os.getenv('OPEN_AI')


# Update the global settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="FacebookAI/xlm-roberta-base"

)

llm = Groq(model='llama3-70b-8192', api_key=groq_api_key)

Settings.llm = llm

chroma_client = chromadb.EphemeralClient()
# chroma_client.delete_collection('uploaded-docs')
chroma_collection = chroma_client.create_collection('temporary-docs')


def process_documents(files, progress=None):
    total_files = len(files)
    documents = []
    for i, file in enumerate(files):
        doc = SimpleDirectoryReader(input_files=[file]).load_data()
        documents.extend(doc)
        if progress is not None:
            progress((i + 1) / total_files, f"Processing file {i + 1}/{total_files}")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
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
    
    progress_bar = gr.Progress()
    process_status = gr.Textbox(label="Status", value="")
    
    chatbot_interface = gr.Chatbot(height=300)
    msg = gr.Textbox(placeholder="Fammi delle domande sui tuoi documenti. Non ti deluderò!")
    clear = gr.Button("Clear")

    def process_and_update(files):
        if files:
            new_index = process_documents([file.name for file in files], progress=progress_bar)
            return new_index, "Documents processed successfully!"
        return None, "No files uploaded."

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, index):
        bot_message = chatbot(history[-1][0], history, index)
        history[-1][1] = bot_message
        return history

    upload_button.click(
        process_and_update,
        inputs=[file_output],
        outputs=[index, process_status]
    )

    msg.submit(user, [msg, chatbot_interface], [msg, chatbot_interface], queue=False).then(
        bot, [chatbot_interface, index], chatbot_interface
    )

    clear.click(lambda: None, None, chatbot_interface, queue=False)

demo.launch()



# print(Settings.embed_model)
# print(Settings.llm)

