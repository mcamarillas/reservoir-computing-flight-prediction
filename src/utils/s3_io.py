import boto3
import io
import numpy as np
import json 
import pandas as pd
import torch
import os

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

s3_client = boto3.client('s3')

def read_csv_from_s3(bucket: str, key: str):

    try:
        key = key.replace(os.sep, '/')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response['Body'])
        print(f"Successfully loaded CSV from S3: {bucket}/{key}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading CSV from S3: {e}")
        return None

def read_json_from_s3(bucket: str, key: str):
    try:
        key = key.replace(os.sep, '/')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Successfully loaded JSON from S3: {bucket}/{key}")
        return data
    except Exception as e:
        print(f"Error reading JSON from S3 {bucket}-{key}: {e}")
        return None

def read_npy_from_s3(bucket: str, key: str):
    try:
        key = key.replace(os.sep, '/')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = np.load(io.BytesIO(response['Body'].read()))
        print(f"Successfully loaded NPY from S3. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error reading NPY from S3: {e}")
        return None

def load_torch_model_from_s3(model: torch.nn.Module, bucket: str, key: str, device: str = 'cpu'):
    try:
        key = key.replace(os.sep, '/')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        state_dict = torch.load(io.BytesIO(response['Body'].read()), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"Successfully loaded PyTorch model from S3: {bucket}/{key}")
        return model
    except Exception as e:
        print(f"Error loading Torch model from S3: {e}")
        return None

def write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str):
    try:
        key = key.replace(os.sep, '/')
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        print(f"Successfully saved CSV to S3: {bucket}/{key}")
    except Exception as e:
        print(f"Error saving CSV to S3: {e}")

def write_json_to_s3(data: dict, bucket: str, key: str):
    try:
        key = key.replace(os.sep, '/')
        json_buffer = json.dumps(data, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        s3_client.put_object(Bucket=bucket, Key=key, Body=json_buffer)
        print(f"Successfully saved JSON to S3: {bucket}/{key}")
    except Exception as e:
        print(f"Error saving JSON to S3: {e}")

def write_npy_to_s3(data: np.ndarray, bucket: str, key: str):
    try:
        key = key.replace(os.sep, '/')
        buffer = io.BytesIO()
        np.save(buffer, data)
        s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        print(f"Successfully saved NPY to S3: {bucket}/{key}")
    except Exception as e:
        print(f"Error saving NPY to S3: {e}")

def write_torch_model_to_s3(model: torch.nn.Module, bucket: str, key: str):
    try:
        key = key.replace(os.sep, '/')
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        print(f"Successfully saved PyTorch model to S3: {bucket}/{key}")
    except Exception as e:
        print(f"Error saving Torch model to S3: {e}")

def save_plot_to_s3(fig, bucket: str, key: str):
    try:
        key = key.replace(os.sep, '/')
        
        if not key.lower().endswith('.png'):
            key += '.png'
            
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        s3_client.put_object(
            Bucket=bucket, 
            Key=key, 
            Body=buffer.getvalue(),
            ContentType='image/png'
        )
        print(f"Successfully saved Plot to S3: {bucket}/{key}")
    except Exception as e:
        print(f"Error saving Plot to S3: {e}")