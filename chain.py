import os
import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from utils import load_config
# Load configuration variables
config=load_config()


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['textsplitter']['chunk_size'], 
        chunk_overlap=config['textsplitter']['chunk_overlap'],
        separators=config['textsplitter']['separators']
        )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings(       
        model_name=config['embeddings']['model_name'], 
        model_kwargs=config['embeddings']['model_kwargs'],
        encode_kwargs=config['embeddings']['encode_kwargs']
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get base model and ollama model
def get_model(model_processing,model_selection):
    model_params = {
        "temperature": config['model']['temperature'],
        "max_tokens": config['model']['max_tokens'],
        "top_p": config['model']['top_p'],
        "n_ctx": config['model']['n_ctx'],
        "verbose": config['model']['verbose']
    }
    
    if model_processing == "GPU":
        model_params.update({
            "n_gpu_layers": config['model']['n_gpu_layers'],
            "n_threads": config['model']['_threads'],
            "n_batch": config['model']['n_batch']
        })

    if model_selection == "FineTuned - Gemma:2B":
        model = LlamaCpp(model_path=config['model']['model_path'], **model_params)
    else:
        model = Ollama(model=config['model']['ollama'])
    return model


# Function to load conversational chain for question answering
def get_conversational_chain(model_processing,model_selection):
    prompt_template = """
    You are a wonderful assistant who has a great understanding of Vector documents and has advanced search functionalities.Please answer the question as clearly as possible from the provided context.
    If the answer is not available in the context, simply say, "Answer is not available in the Context"; please don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = get_model(model_processing,model_selection)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to handle user input and generate response
def user_input(user_question,model_processing,model_selection):
    embeddings = HuggingFaceBgeEmbeddings(       
        model_name=config['embeddings']['model_name'], 
        model_kwargs=config['embeddings']['model_kwargs'],
        encode_kwargs=config['embeddings']['encode_kwargs']
    )
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(model_processing,model_selection)   
    response = chain.invoke(
        {"input_documents":docs,
        "question": user_question},
        return_only_outputs=True
    )   
    return response["output_text"]


# Function to split url text into chunks
def get_url_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=config['textsplitter']['chunk_size'], 
        chunk_overlap=config['textsplitter']['chunk_overlap'],
        )
    chunks = text_splitter.split_documents(text)
    return chunks


# Function to create vector store from text chunks for URL
def get_url_vector_store(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings(       
        model_name=config['embeddings']['model_name'], 
        model_kwargs=config['embeddings']['model_kwargs'],
        encode_kwargs=config['embeddings']['encode_kwargs']
    )
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")