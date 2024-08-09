import os 
import pandas as pd
from sqlalchemy import create_engine,inspect

class PrepareSQLFromFlatFiles:
    def __init__(self,files_dir):
        self.files_directory= files_dir
        self.files_list = os.listdir(files_dir)
        db_path="docs/csv_xlsx.db"
        db_path=f"sqlite:///{db_path}"
        self.engine = create_engine(db_path)
        print("number of csv files:", len(self.files_list))


    def _prepare_db(self):
        for file in self.files_list:
            full_file_path= os.path.join(self.files_directory,file)
            file_name,files_extension= os.path.splitext(file)
            if files_extension==".csv":
                df=pd.read_csv(full_file_path)
                df.to_sql(file_name,self.engine,index=False)
            elif files_extension==".xlsx":
                df=pd.read_excel(full_file_path)
                df.to_sql(file_name,self.engine,index=False)
            print(f"csv file:{file_name} saved to db")

    def _validate_db(self):
        inspector = inspect(self.engine)
        table_names=inspector.get_table_names()
        print("Avaliable table names created in sql db:", table_names)

    def run_pipeline(self):
        if len(self.files_list)<2:
            self._prepare_db()
        self._validate_db()
