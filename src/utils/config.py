import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

bucket = os.getenv('S3_BUCKET_NAME')

class ModelType(Enum): 
    RESERVOIR_COMPUTING = 0,
    LSTM = 1,
    LGBM = 2