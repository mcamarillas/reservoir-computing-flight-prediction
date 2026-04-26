from enum import Enum

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb

class ModelType(Enum): 
    RESERVOIR_COMPUTING = 0,
    LSTM = 1,
    LGBM = 2

class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FlightDataPreparer:    
    def __init__(self, target_column: str,  window_size: int = 12, required_length: int = 100):
        self.target_column = target_column
        self.window_size = window_size
        self.required_length = required_length

    def transform(self, df: pd.DataFrame, model_type: ModelType):
        match model_type:
            case ModelType.RESERVOIR_COMPUTING:
                return self._prepare_reservoir_computing_input_data(df)
            case ModelType.LSTM:
                return self._prepare_lstm_input_data(df)
            case ModelType.LGBM:
                raise Exception("LGBM not implemented")
            case _:
                raise Exception(f"{model_type} not implemented")


    def _prepare_reservoir_computing_input_data(self, df: pd.DataFrame):
        icao_group_by = df.groupby('icao24')
        icao_list = []
        X_list = []
        y_list = []

        for icao, icao_slice in icao_group_by:
            if len(icao_slice) < self.required_length:
                continue
            icao_slice = icao_slice.drop(columns=["icao24", "time"])

            X_list.append(icao_slice.to_numpy()) 
            y_list.append(icao_slice[self.target_column].to_numpy().reshape(-1, 1))
            icao_list.append(icao)
            
        X_list = [x[:-self.window_size] for x in X_list]
        y_list = [y[self.window_size:] for y in y_list]
        return X_list, y_list, icao_list

    def _prepare_lstm_input_data(self, df: pd.DataFrame):
        X, y, icao_list = self._prepare_windowed_data(df)
        dataset = FlightDataset(X, y)
        return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True), X.shape[2], icao_list

    def prepare_lgbm_input_data(self, X: pd.DataFrame):
        """
        X, y, icao_counts = self._prepare_windowed_data(
            X, 
            target_column=self.target_column, 
            window_size=self.window_size, 
            required_length=self.required_length
        )
        X_lgbm = X.reshape(X.shape[0], -1)
        y_lgbm = np.array(y, dtype=np.float32).flatten()

        train_set = lgb.Dataset(X_lgbm, label=y_lgbm)
        test_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
        """
        return False

    def _prepare_windowed_data(self, X: pd.DataFrame):
        icao_group_by = X.groupby('icao24')
        X_windows = []
        y_targets = []
        icao_counts = []

        for icao, icao_slice in icao_group_by:
            icao_slice = icao_slice.drop(columns=["icao24", "time"])
            
            data = icao_slice.to_numpy(dtype=np.float32)
            velocity_idx = icao_slice.columns.get_loc(self.target_column)
            if len(data) < self.required_length:
                continue
            num_windows = len(data) - 2 * self.window_size
            icao_counts.append((icao, num_windows))
            for i in range(num_windows):
                X_windows.append(data[i : i + self.window_size])
                y_targets.append(data[i + 2 * self.window_size, velocity_idx])

        return np.array(X_windows), np.array(y_targets), icao_counts
