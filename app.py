import time
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_core.messages import AIMessage, HumanMessage
from chain import *
from src.chatbot import get_chatbot_response
from src.pdf import get_pdf_text
from src.website import get_website_text
from src.youtube import get_youtube_text

st.set_page_config(page_title="Multimodal Rag Application", page_icon="🤖", layout="wide")

with st.sidebar:
    selected = option_menu('Multimodal Rag Application',
                           ['ChatBot','PDF','Website','YouTube'],
                           menu_icon='none', 
                           icons=['robot','filetype-pdf', 'browser-chrome','youtube'],
                           default_index=0
                           )
    model_processing = st.radio("Model processing",options=["CPU","GPU"])
    model_selection = st.radio("Model Selection",options=["FineTuned - Gemma:2B","Ollama - Gemma:2B"])

    def clear_cache():
        keys = list(st.session_state.keys())
        for key in keys:
            st.session_state.pop(key)
    st.button('New Chat', on_click=clear_cache)


if selected == 'ChatBot':
    st.title(":red[Interactive Assistant] - :blue[ChatBot]")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="“Hello 👋   How may I assist you today?”")]
    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.write(message.content)
    user_query = st.chat_input("Message ChatBot...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            st_time = time.time()
            response = st.write_stream(get_chatbot_response(user_query, st.session_state.chat_history,model_processing,model_selection))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.write('Response Time = ',(time.time()-st_time))

elif selected in ['PDF', 'Website', 'YouTube']:
    title_dict = {'PDF': 'PDF', 'Website': 'Website', 'YouTube': 'YouTube'}
    st.title(f":red[Multimodal Rag Application] - :blue[{title_dict[selected]}]")
    user_input_placeholder = "Enter Your Question Here"
    if selected == 'PDF':
        pdf_docs = st.file_uploader("Upload your PDF Files ", accept_multiple_files=True, type="pdf")
        get_text_func = get_pdf_text
    elif selected == 'Website':
        URLS = st.text_input(" Enter URL Here")
        get_text_func = get_website_text
    else:
        URLS = st.text_input(" Enter URL Here")
        get_text_func = get_youtube_text

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content":"“Hello 👋   How may I assist you today?”"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(user_input_placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Generating answer..."):
            raw_text = get_text_func(pdf_docs if selected == 'PDF' else URLS)
            text_chunks = get_text_chunks(raw_text) if selected == 'PDF' else get_url_text_chunks(raw_text)
            if selected == 'PDF':
                get_vector_store(text_chunks)
            elif selected in ['Website', 'YouTube']:
                get_url_vector_store(text_chunks)
            st_time = time.time()
            response = user_input(prompt,model_processing,model_selection)
        with st.chat_message("assistant"):
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
            st.write('Response Time = ',(time.time()-st_time))