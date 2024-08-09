import os
import pandas as pd
import cohere
import chromadb

class PrepareVectorDBfomFLATFILES:
    def __init__(self, file_path, path_to_db):
        self.cohere_client= cohere.Client(api_key=os.getenv("cohere_api_key"))
        self.embedding_model_name= "embed-english-v3.0"
        self.file_path= file_path
        self.path_to_db=path_to_db
        self.chroma_client= chromadb.PersistentClient(path=str(path_to_db))

    def run_pipeline(self):
        self.df,self.file_name= self._load_dataframe(file_path=self.file_path)
        self.docs,self.metadata,self.ids,self.embeddings=self._prepare_data_for_injestion(self.df,self.file_name)
        self._inject_data_to_chromadb()



    def _load_dataframe(self,file_path):
        file_name_extensions= os.path.basename(file_path)
        file_name, file_extension= os.path.splitext(file_name_extensions)
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        return df,file_name


    def _prepare_data_for_injestion(self,df, file_path):
        docs=[]
        metadata=[]
        ids=[]
        embeddings=[]
        for index, row in df.iterrows():
            output_str=""
            for col in df.columns:
                output_str += f"{col}: \t{str(row[col])}"

            docs.append(output_str)
            response= self.cohere_client.embed(texts=docs,
            model=self.embedding_model_name,input_type="search_query")
            embeddings.append(response.embeddings[0])
            metadata.append({"source":file_path})
            ids.append(f"id{index}")
        return docs,metadata,ids,embeddings
    
    def _inject_data_to_chromadb(self):
        collection_name_with_extension=os.path.basename(self.file_path)
        collection_name, collection_extension= os.path.splitext(collection_name_with_extension)
        print(collection_name)
        print(self.chroma_client.list_collections())
        existing_collections = [col.name for col in self.chroma_client.list_collections()]
        if collection_name not in existing_collections:
            # Create a new collection if it doesn't exist
            collection = self.chroma_client.create_collection(name=collection_name)
            collection.add(documents=self.docs, metadatas=self.metadata, embeddings=self.embeddings, ids=self.ids)
            print("Data stored into ChromaDB")
        else:
            # Use the existing collection
            collection = self.chroma_client.get_collection(name=collection_name)
            print("Vector DB count")
            print(collection.count())
            return collection