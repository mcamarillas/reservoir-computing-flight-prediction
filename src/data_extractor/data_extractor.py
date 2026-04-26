import pandas as pd
import json
import os
import numpy as np
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer

from utils.io import read_csv_file, read_json_file, write_csv_file
from utils.logger import get_logger

logger = get_logger("FlightDataExtractor")

class FlightDataExtractor:
    def __init__(self, metadata_file: str, dataset_suffix: str, base_path: str, trans_path: str):
        self.metadata_file = metadata_file
        self.suffix = dataset_suffix
        self.base_path = base_path
        self.trans_path = trans_path
        
        self.metadata = read_json_file(self.metadata_file)
        self.preprocessor = None
        self.transformation_map = {}

        self.SPLIT_FRAC = 0.8
        self.FORECAST_WINDOW = 12
        self.MIN_LENGTH = self.FORECAST_WINDOW + 2

    def run_pipeline(self, file_list: list[str]):
        """
        Executes the full ETL and Preprocessing flow
        """
        logger.info(f"Reading metadata from {self.metadata_file}")
        
        df = self._load_and_clean_data(file_list)

        df = self._split_icao_by_gaps(df)
        df = self._filter_flights_by_nan(df)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        df = self._interpolate_nan_values(df, numeric_cols)
        
        cols_to_diff = self._get_cols_to_diff()
        df = self._calculate_column_diffs(df, cols_to_diff)

        train_df, test_df = self._train_test_split(df)   
        transformed_train, transformed_test = self._transform_data(train_df, test_df)
        
        self._save_results(transformed_train, transformed_test)

    def _load_and_clean_data(self, file_list: list[str]) -> pd.DataFrame:
        all_frames = []
        var_types = self._get_variable_types()
        cols_to_drop = self._get_non_essential_columns()
        
        for file_path in file_list:
            try:
                logger.info(f"Processing {file_path}...")
                df = read_csv_file(file_path)
                df = self._cast_data(df, var_types)
                df = df.drop(columns=cols_to_drop)
                all_frames.append(df)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        return pd.concat(all_frames, ignore_index=True)

    def _split_icao_by_gaps(self, df: pd.DataFrame, interval: int = 10) -> pd.DataFrame:
        if df.empty: return df
        time_diffs = df.groupby('icao24')['time'].diff()
        is_gap = (time_diffs.fillna(interval) != interval)
        sequence_grp = is_gap.groupby(df['icao24']).cumsum()
        df['icao24'] = df['icao24'].astype(str) + '.' + sequence_grp.astype(str)
        return df

    def _filter_flights_by_nan(self, df: pd.DataFrame, threshold_pct: float = 10.0) -> pd.DataFrame:
        any_nan_mask = df.isnull().any(axis=1)
        nan_stats = any_nan_mask.groupby(df['icao24']).agg(['count', 'sum'])
        nan_stats['nan_pct'] = (nan_stats['sum'] / nan_stats['count']) * 100
        valid_segments = nan_stats[nan_stats['nan_pct'] <= threshold_pct].index
        return df[df['icao24'].isin(valid_segments)].copy()

    def _interpolate_nan_values(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        grouped = df.groupby('icao24')[cols]
        ffill_vals = grouped.ffill()
        ffill_diffs = grouped.diff().groupby(df['icao24']).ffill()

        for col in cols:
            is_nan = df[col].isna()
            if not is_nan.any(): continue
            
            block_id = (~is_nan).cumsum()
            nan_count = df.groupby(['icao24', block_id]).cumcount()
            projected = ffill_vals[col] + (nan_count * ffill_diffs[col])
            df.loc[is_nan, col] = projected[is_nan]

        return df.dropna().reset_index(drop=True)

    def _calculate_column_diffs(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        diffed_values = df.groupby('icao24')[cols].diff()
        df[[f"{col}_diff" for col in cols]] = diffed_values
        return df.dropna().reset_index(drop=True)

    def _train_test_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        icao_list = [
            key for key, group in df.groupby('icao24') 
            if not group.isnull().values.any() and len(group) > self.MIN_LENGTH
        ]

        train_size = int(len(icao_list) * self.SPLIT_FRAC)
        train_list, test_list = icao_list[:train_size], icao_list[train_size:]
        
        return df[df["icao24"].isin(train_list)], df[df["icao24"].isin(test_list)]

    def _transform_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.transformation_map = self._get_cols_by_transformation()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("std", StandardScaler(), self.transformation_map.get("std", [])),
                ("minmax", MinMaxScaler(), self.transformation_map.get("min-max", [])),
                ("maxabs", MaxAbsScaler(), self.transformation_map.get("max-abs", [])),
                ("sin", FunctionTransformer(lambda x: np.sin(np.deg2rad(x))), self.transformation_map.get("cyclic", [])),
                ("cos", FunctionTransformer(lambda x: np.cos(np.deg2rad(x))), self.transformation_map.get("cyclic", [])),
                ("keep", "passthrough", self.transformation_map.get("passthrough", []))
            ],
            remainder="drop"
        )

        self.preprocessor.set_output(transform="pandas")
        train_scaled = self.preprocessor.fit_transform(train_df)
        test_scaled = self.preprocessor.transform(test_df)

        train_scaled.columns = [c.split('__')[-1] for c in train_scaled.columns]
        test_scaled.columns = [c.split('__')[-1] for c in test_scaled.columns]

        self._save_transformation_metadata()
        return train_scaled, test_scaled

    def _get_variable_types(self) -> dict:
        all_vars = {**self.metadata["numerical_variables"], **self.metadata["categorical_variables"]}
        return {name: self._type_str_to_type(info["type"]) for name, info in all_vars.items()}

    def _type_str_to_type(self, type_str: str):
        mapping = {"integer": "Int64", "float": float, "string": str}
        if type_str not in mapping:
            raise ValueError(f"Type {type_str} not supported")
        return mapping[type_str]

    def _get_non_essential_columns(self) -> list:
        control_vars = {self.metadata.get("step_variable"), self.metadata.get("group_by_variable")}
        all_vars = {**self.metadata["numerical_variables"], **self.metadata["categorical_variables"]}
        return [n for n, info in all_vars.items() if not info.get("include", False) and n not in control_vars]

    def _get_cols_to_diff(self) -> list:
        num_vars = self.metadata.get("numerical_variables", {})
        return [n for n, info in num_vars.items() if info.get("calculate_diff", False)]

    def _get_cols_by_transformation(self) -> dict:
        groups = defaultdict(list)
        all_vars = {**self.metadata["numerical_variables"], **self.metadata["categorical_variables"]}
        for name, info in all_vars.items():
            trans = info.get("transformation", "")
            if not trans: continue
            groups[trans].append(name)
            if info.get("calculate_diff", False):
                groups[trans].append(f"{name}_diff")
        return dict(groups)

    def _cast_data(self, df: pd.DataFrame, var_types: dict) -> pd.DataFrame:
        for variable, var_type in var_types.items():
            if variable not in df:
                raise KeyError(f"Error: The key {variable} is not in the dataset")
            try:
                df[variable] = df[variable].astype(var_type)
            except Exception as e:
                print(f"Error: The variable {variable} casting failed: {e}")
        return df

    def _save_transformation_metadata(self):
        meta = {}
        for key, name in [('std', 'std'), ('min-max', 'minmax'), ('max-abs', 'maxabs')]:
            if name in self.preprocessor.named_transformers_:
                scaler = self.preprocessor.named_transformers_[name]
                for i, col in enumerate(self.transformation_map.get(key, [])):
                    if name == 'std': meta[col] = {"scale": scaler.scale_[i], "mean": scaler.mean_[i]}
                    elif name == 'minmax': meta[col] = {"min": scaler.data_min_[i], "max": scaler.data_max_[i]}
                    elif name == 'maxabs': meta[col] = {"max_abs": scaler.max_abs_[i]}

        out_path = os.path.join(self.trans_path, f"dataset_{self.suffix}_transformations.json")
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=4)

    def _save_results(self, train: pd.DataFrame, test: pd.DataFrame):
        write_csv_file(train, os.path.join(self.base_path, "train", f"data_{self.suffix}.csv"))
        write_csv_file(test, os.path.join(self.base_path, "test", f"data_{self.suffix}.csv"))

