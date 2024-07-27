import os
from datetime import datetime

#langchain library
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


from src.utils.helper import save_uploadedfile,PDF_loader,generate_response
import streamlit as st


st.set_page_config(page_title="Chat with PDF", page_icon="🎁")



if not os.path.exists('./docs'):
    os.makedirs('./docs')


tab1, tab2 = st.tabs(
    ["💬 Chat with PDF", "Relevant documents metadata"]
)


with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; color: #007BFF;'>Upload PDF</h2>",
        unsafe_allow_html=True,
    )
    cohere_api_key = st.text_input("Enter your Cohere API key", type="password")
    if cohere_api_key:
        try:
            test_model = ChatCohere(model="command", cohere_api_key=cohere_api_key)
            response = test_model.invoke("Hello")
            if "error" in response:
                raise Exception("Invalid API key🚩")
            else:
                st.success("Sucessfully SET API Token ✅")
            
        except Exception as e:
            st.error(f"Error: {str(e)} - Please enter a correct Cohere API key.❌")
            st.stop()
    else:
        st.error("Please enter your Cohere API key to proceed.")

 
    uploaded_files = st.file_uploader("Choose a file", type=["pdf"],accept_multiple_files=True)

    with st.expander("Adjust Settings 🛠"):
        temp_r = st.slider("Temperature", 0.3, 0.9, 0.10, 0.6)
        chunksize = st.slider("Chunk Size for Splitting Document", 256, 400, 1024, 10)
        clear_button = st.button("Clear Conversation", key="clear")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunksize, chunk_overlap=50, separators=[" ", ",", "\n"]
)



if cohere_api_key:
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key,
    )
    model_cohere = ChatCohere(
        model="command",
        temperature=temp_r,
        cohere_api_key=cohere_api_key,
    )



if uploaded_files is not None and cohere_api_key:
    saved_files = []
    for uploaded_file in uploaded_files:
        save_uploadedfile(uploaded_file)
        saved_files.append("docs/" + uploaded_file.name)
        file_size = os.path.getsize(f"docs/{uploaded_file.name}") / (1024 * 1024)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Uploaded PDF: {file_size} MB")
        st.markdown(
            "<h3 style='text-align: center;'>Now You Are Chatting With "
            + uploaded_file.name
            + "</h3>",
            unsafe_allow_html=True,
            )
    qa = PDF_loader(saved_files,embeddings=embeddings,text_splitter=text_splitter,model=model_cohere,cohere_api_key=cohere_api_key)
    


prompt = st.chat_input("Docs wont say hi, so Start the conversation with your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if uploaded_file is not None and cohere_api_key:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response=""
            while not full_response:
                with st.spinner("Thinking..."):
                    Output, source_documents = generate_response(prompt, qa)
                    full_response= Output if Output else "Failed to get the response"
                    tab2.markdown(
                        "<h3 style='text-align: center;'>Relevant Documents Metadata</h3>",
                        unsafe_allow_html=True,
                    )
                    tab2.write(source_documents)

                fr=""
                for i in str(full_response):
                    import time
                    time.sleep(0.03)
                    fr+= i
                    message_placeholder.write(f"{fr} + |")
                message_placeholder.write(f"{full_response}")    
            st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write(
                "Please go ahead and upload the PDF in the sidebar, it would be great to have it there and make sure API key Entered"
            )
            st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Please go ahead and upload the PDF in the sidebar, it would be great to have it there and make sure API key Entered",
            })

if clear_button:
    st.session_state.messages = []