import pandas as pd
import os
from datetime import datetime

class Logger:
    def __init__(self):
        self.log = {}
    def add_log(self,feature_name,value):
        self.log[feature_name] = value       
    def fill_missing_values(self,data: dict) -> dict:
        max_len = max([len(v) for v in data.values()])
        for key in data:
            if len(data[key]) < max_len:
                data[key] += [None] * (max_len - len(data[key]))
        return data
    def write_to_csv(self, file_name):
        filled_data = self.fill_missing_values(self.log)
        df = pd.DataFrame(filled_data)
        df.to_csv(file_name, index=False)
logger = Logger()
logger.add_log("accuracy", [0.9])
logger.add_log("loss", [0.1,0.2,0.3])
logger.log['loss'].append(0.9)
logger.write_to_csv("logs.csv")
# Read logs.csv and print it 
df = pd.read_csv("logs.csv")
print(type(df['loss'][0]))