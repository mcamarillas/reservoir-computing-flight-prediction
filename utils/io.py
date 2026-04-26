import pandas as pd
import json
import os
import numpy as np
import torch 

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def read_csv_file(file_path: str):
    """
    Reads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None

def read_json_file(file_path: str):
    """
    Reads a JSON file into a pandas DataFrame.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON as a dictionary.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except ValueError:
        print(f"Error: The file at '{file_path}' is not a valid JSON or has an incompatible format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None

def read_npy_file(file_path: str):
    """
    Reads a NumPy array from a .npy file.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' does not exist.")
            return None
        data = np.load(file_path)
        print(f"Successfully loaded NPY from {file_path}. Shape: {data.shape}")
        return data
    except PermissionError:
        print(f"Error: Permission denied when trying to read '{file_path}'.")
    except ValueError:
        print(f"Error: The file at '{file_path}' is corrupted or not a valid .npy file.")
    except Exception as e:
        print(f"An unexpected error occurred while loading: {e}")
        return None

def load_torch_model(model: torch.nn.Module, file_path: str, device: str = 'cpu'):
    """
    Loads the PyTorch model state dictionary into an existing instance.
    """
    try:
        state_dict = torch.load(file_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"Successfully loaded PyTorch model from {file_path} to {device}")
        return model
    except RuntimeError as e:
        print(f"Error: Architecture mismatch. Could not load state dict: {e}")
    except PermissionError:
        print(f"Error: Permission denied when trying to read '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None

def write_csv_file(df: pd.DataFrame, file_path: str):
    """
    Writes a pandas DataFrame to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Successfully saved CSV to {file_path}. Shape: {df.shape}")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while saving: {e}")

def write_json_file(data: dict, file_path: str):
    """
    Writes a dictionary to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"Successfully saved JSON to {file_path}. Items: {len(data)}")
    except TypeError:
        print(f"Error: The data provided is not JSON serializable.")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while saving: {e}")

def write_npy_file(data: np.ndarray, file_path: str):
    """
    Writes a NumPy array to a .npy file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, data)
        print(f"Successfully saved NPY to {file_path}. Shape: {data.shape}")
    except AttributeError:
        print(f"Error: The data provided is not a valid NumPy array.")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while saving NPY: {e}")

def write_torch_model(model: torch.nn.Module, file_path: str):
    """
    Saves the PyTorch model state dictionary.
    """
    try:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(model.state_dict(), file_path)
        print(f"Successfully saved PyTorch model to {file_path}")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to '{file_path}'.")
    except AttributeError:
        print("Error: The provided object is not a valid PyTorch model.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the model: {e}")


