import os
from datetime import datetime

#langchain library
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from src.utils.helper_pdf_file import save_uploadedfile,PDF_loader,generate_response
from src.utils.helper_flat_files import PrepareVectorDBfomFLATFILES
from src.utils.helper_sql_db import PrepareSQLFromFlatFiles
import streamlit as st


st.set_page_config(page_title="Chat with Any Database", page_icon="🎁")



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

 
    uploaded_files = st.file_uploader("Choose a file", type=["pdf","csv",'xlsx'],accept_multiple_files=True)

    with st.expander("Adjust Settings 🛠"):
        temp_r = st.slider("Temperature", 0.3, 0.9, 0.10, 0.6)
        chunksize = st.slider("Chunk Size for Splitting Document", 256, 400, 1024, 10)
        clear_button = st.button("Clear Conversation", key="clear")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunksize, chunk_overlap=50, separators=[" ", ",", "\n"]
)





if cohere_api_key:
    os.environ["cohere_api_key"]= cohere_api_key
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key,
    )
    model_cohere = ChatCohere(
        model="command",
        temperature=temp_r,
        cohere_api_key=cohere_api_key,
    )

prompt = st.chat_input("Docs wont say hi, so Start the conversation with your documents")

if uploaded_files is not None and cohere_api_key:
    saved_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
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

        
        elif uploaded_file.name.endswith('.csv'):
            save_uploadedfile(uploaded_file)

            approach = st.radio(
                "Select an approach",
                ["Default","RAG approach", "LLM Agent"],
                captions=["Select an approach",
                    "converting csv files to Vector store embedding.",
                    "converting csv files to sql db and uses SQL LLM Agent",
                ],
            )

            if approach == "RAG approach":
                st.write("You have selected rag")
               
                RAG_pipeline=PrepareVectorDBfomFLATFILES(file_path="docs/" + uploaded_file.name,path_to_db=f"./docs/db_{os.path.basename('docs/' + uploaded_file.name).rsplit('.')[0]}")    
                RAG_pipeline.run_pipeline()

            elif approach== "LLM Agent":
                st.write("You have selected LLM agent")        
                llm_agent_pipeline=PrepareSQLFromFlatFiles('docs')  
                llm_agent_pipeline.run_pipeline()          
            
            else:
                st.write("please select a button.")



    




if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    for uploaded_file in uploaded_files:
        if uploaded_file is not None and cohere_api_key and uploaded_file.name.endswith('.pdf'):
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
                
        elif uploaded_file is not None and cohere_api_key and uploaded_file.name.endswith('.csv'):
             with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response=''                   
                while not full_response:
                    with st.spinner("Thinking..."):
                        if approach=="RAG approach":
                            data_instance=PrepareVectorDBfomFLATFILES(file_path="docs/" + uploaded_file.name,path_to_db=f"./docs/db_{os.path.basename('docs/' + uploaded_file.name).rsplit('.')[0]}")
                            response=data_instance.cohere_client.embed(texts=[prompt],
                                    model=data_instance.embedding_model_name,input_type="search_query")
                                
                            query_embeddings= response.embeddings[0]
                            vector_db=data_instance._inject_data_to_chromadb()
                            results=vector_db.query(query_embeddings=query_embeddings,n_results=1)
                            print("similarity search based retrieval", results)

                            user_query= f"User's question {prompt} \n\n Search results: \n {results}"
                            messages= [
                                {"role": "system","content":"You will recieve the user's question along with the search results of that question over a database. Give the user the proper answer"},
                                {"role":"user", "content": str(user_query)}
                            ]
                            llm_response= model_cohere.invoke(messages)
                            full_response= llm_response.content

                        if approach=="LLM Agent":
                            engine=create_engine(f"sqlite:///docs/csv_xlsx.db")
                            db = SQLDatabase(engine=engine)
                            print(db.dialect)
                            print(db.get_usable_table_names())
                            agent_executor = create_sql_agent(
                                    llm=model_cohere,
                                    toolkit=SQLDatabaseToolkit(db=db, llm=model_cohere),
                                    verbose=True,
                                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                )

                            response = agent_executor.run(prompt)
                            print(response)
                            full_response = response["output"]
                            print("full_resp",full_response)
                        print("full_resp1",full_response)
                    print("full_resp2",full_response)
                    fr=''
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