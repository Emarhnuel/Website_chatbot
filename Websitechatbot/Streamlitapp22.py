import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain.chat_models import ChatOpenAI
from HTMLTemplate import css, user_template, bot_template
from dotenv import load_dotenv

# Set your API keys from environment variables
google_api_key = os.getenv('GOOGLE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')


# Load environment variables
load_dotenv()

# Function to load data from URLs using Beautiful Soup
def load_urls(urls):
    try:
        all_text = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        for url in urls:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                texts = soup.stripped_strings
                all_text.append(' '.join(texts))
            else:
                st.error(f"Failed to load {url}: HTTP {response.status_code}")
                continue
        if not all_text:
            st.error("No data returned from the URLs.")
            return None
        return all_text
    except Exception as e:
        st.error(f"Error occurred while loading data from URLs: {e}")
        return None

# Function to get text chunks from data
def get_text_chunks(data):
    all_chunks = []
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=200, length_function=len)
    for text in data:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks

# Function to get vector store
def get_vector_store(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

# Function to get conversation chain
def get_conversation_chain(vectorstore):
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, convert_system_message_to_human=True)
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

# Function to handle user input
def handle_user_input(user_question):
    max_tokens = 32000
    truncated_input = user_question[:max_tokens]

    response = st.session_state.conversation({'question': truncated_input})
    st.session_state.chat_history = response['chat_history']

# Function to display chat history
def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Function to clear chat history
def clear_chat_history():
    st.session_state.chat_history = []

# Main function to run the app
def main():
    st.set_page_config(page_title="Chatbot for Your Own Website", page_icon=":chatbot:")
    st.markdown(css, unsafe_allow_html=True)
    st.header("Chatbot For Your Own Website 🌐")

    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Moved the user input section below the messages display
    if st.session_state.messages:
        for message in st.session_state.messages:
            role = "assistant" if message["role"] == "assistant" else "user"
            # Use st.container or similar to create message blocks
            with st.container():
                st.write(f"{role.capitalize()}: {message['content']}")




    with st.sidebar:
        st.title("LLM Chatapp using Gemini Pro and Langchain 🔗")

        user_url = st.text_input("Enter a website URL to chat with", key="user_url")
        if st.button("Start"):
            if user_url:
                with st.spinner("Processing"):
                    data = load_urls([user_url])
                    if data:
                        text_chunks = get_text_chunks(data)
                        if text_chunks:
                            vectorstore = get_vector_store(text_chunks)
                            if vectorstore:
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                st.success("Chat is ready!")
                            else:
                                st.error("Failed to create a vector store.")
                        else:
                            st.error("Text chunks could not be processed.")
                    else:
                        st.error("No data to process. Please check the URL and try again.")
        if st.button("Clear Chat"):
            clear_chat_history()
            st.rerun()

        # Contact details
        st.markdown("- This chatbot was created by Emmanuel Ezeokeke")
        st.markdown("- Contact him [LinkedIn Profile](https://www.linkedin.com/in/emma-ezeokeke/)")

    user_question = st.chat_input("Please ask any question", key="user_question")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    # Display chat history
    if 'chat_history' in st.session_state:
        display_chat_history()

if __name__ == '__main__':
    main()




