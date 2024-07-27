import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain                       
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever


def save_uploadedfile(uploadedfile):
    with open(os.path.join("docs", uploadedfile.name),'wb') as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Saved File")


def PDF_loader(documents,embeddings,text_splitter,model,cohere_api_key):
    all_docs = []
    for document in documents:
        db_path=f"./docs/db_{os.path.basename(document).rsplit('.')[0]}"
        if os.path.exists(db_path):
            print("using Cached one")
            db_chroma= Chroma(embedding_function= embeddings,persist_directory=db_path)
            retriever=db_chroma.as_retriever()
        else:
            loader= PyPDFLoader(document)
            pages=loader.load_and_split()
            docs= text_splitter.split_documents(pages)
            db_chroma= Chroma.from_documents(docs,embeddings,persist_directory=db_path)
            retriever= db_chroma.as_retriever()
        all_docs.append(retriever)

    combined_retriever=MergerRetriever(retrievers=all_docs)
    compressor= CohereRerank(cohere_api_key=cohere_api_key)
    compression_retriever= ContextualCompressionRetriever(base_compressor=compressor, base_retriever=combined_retriever)
    prompt_template = """You are an AI chatbot that helps users chat with PDF documents.
        Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you find the answer, write the answer in a Elegant way and add the list of sources that are **directly** used to derive the answer.
        Example:
        The Answer is derived from[1] this page
        [1] Source_ Page:PageNumber

        {context}

        Question: {question}
        Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nPageNumber:{page}\nsource:{source}",)
    llm_chain = LLMChain(
            llm=model, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=False
        )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
        verbose=False,
        )

    qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            callbacks=None,
            verbose=False,
            retriever=compression_retriever,
            return_source_documents=True,
        )

    return qa



def generate_response(query, qa):
    result = qa.invoke({"query": query})
    return result["result"], result['source_documents']

# def extract_reference_answer(source_documents):
#     # Extract and concatenate text from source documents
#     print(source_documents)
#     reference_texts = [doc for doc in source_documents]
#     ground_truth=[doc.page_content for doc in reference_texts]
#     return " ".join(ground_truth)