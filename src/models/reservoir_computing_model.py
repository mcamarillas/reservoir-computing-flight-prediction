import time
import os
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import IPReservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli

from src.hpt.epsilon_greedy_search import EpsilonGreedyReservoirHPSearch_R2
from src.hpt.hp_visualizer import visualize_search
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from src.utils.s3_io import read_json_from_s3, read_npy_from_s3, write_json_to_s3, write_npy_to_s3
from src.utils.logger import get_logger
from src.utils.config import bucket

logger = get_logger("IPReservoirComputingModel")

class IPReservoirComputingModel():
    def __init__(self, name, params=None):
        self.name = name
        self.version = str(int(time.time()))
        self.params = params or {}
        rpy.set_seed(self.params.get("seed", 33))
        self.warmup_steps = self.params.get('warmup', 100)
        
        self.reservoir = IPReservoir(
            units=self.params.get('units', 100),
            sr=self.params.get('sr', 0.9),
            mu=self.params.get('mu', 0.01),
            input_scaling=self.params.get('input_scaling', 1.0),
            learning_rate=self.params.get('learning_rate', 0.01),
            rc_connectivity=self.params.get('connectivity', 0.1),
            input_connectivity=self.params.get('connectivity', 0.1),
            activation=self.params.get('activation', 'sigmoid'),
            epochs=self.params.get('epochs', 1),
            seed=self.params.get('seed', 33)
        )
        self.readout = Ridge(
            ridge=self.params.get('ridge', 1e-6), 
            input_dim=self.params.get('units', 100)
        )

    @staticmethod
    def load_model(name, version, base_path: str = "models"):
        model_path = os.path.join(base_path, name, version)
        
        params = read_json_from_s3(bucket, os.path.join(model_path, "params.json"))
        
        instance = IPReservoirComputingModel(name=name, params=params)
        instance.version = version
        
        weights = read_npy_from_s3(bucket, os.path.join(model_path, "readout_weights.npy"))
        bias = read_npy_from_s3(bucket, os.path.join(model_path, "readout_bias.npy"))
        instance.readout.Wout = weights
        instance.readout.bias = bias
        instance.readout.output_dim = weights.shape[1]
        instance.readout.initialized = True

        return instance
    
    def fit(self, X, y):
        time_start = time.time()
        reservoir_states = self.reservoir.run(X)
        self.readout.fit(reservoir_states, y, warmup=self.warmup_steps)
        self.training_time = time.time() - time_start
        logger.info(f"Training time: {self.training_time}")

    def predict(self, X):
        reservoir_states = self.reservoir.run(X)
        return self.readout.run(reservoir_states)
    
    @staticmethod
    def tune(X_train, y_train, X_test, y_test, name, n_iterations=20, criterion="mape", optimize_objective="minimize", base_path="./data/models"):
        searcher = EpsilonGreedyReservoirHPSearch_R2(X_train, y_train, X_test, y_test, n_iterations=n_iterations,criterion=criterion, optimize_objective=optimize_objective)
        start_time = time.time()
        params, _ = searcher.search(n_iterations=n_iterations)

        logger.info(f"Hyperparameter search time: {time.time() - start_time}")

        hpt_model = IPReservoirComputingModel(name=name, params=params)
        hpt_model.fit(X_train, y_train)
        hpt_model.training_time = time.time() - start_time

        visualize_search(searcher, os.path.join(base_path, hpt_model.name, hpt_model.version, "hpt_report"))
        return hpt_model

    def evaluate(self, y_true, y_pred, eval_function="mape"):
        y_true_flatten = np.concatenate(y_true)
        y_pred_flatten = np.concatenate(y_pred)
        if eval_function == "mape":
            return mean_absolute_percentage_error(y_true_flatten, y_pred_flatten)
        elif eval_function == "r2":
            return r2_score(y_true_flatten, y_pred_flatten)
        else:
            raise Exception(f"Error: The evaluation function {eval_function} is not suported")

    def save_model(self, base_path: str = "models"):
        model_path = os.path.join(base_path, self.name, self.version)
        os.makedirs(model_path, exist_ok=True)
        
        write_json_to_s3(self.params, bucket, os.path.join(model_path, "params.json"))
        
        if self.readout.Wout is not None:
            write_npy_to_s3(self.readout.Wout, bucket, os.path.join(model_path, "readout_weights.npy"))
            write_npy_to_s3(self.readout.bias, bucket, os.path.join(model_path, "readout_bias.npy"))
        else:
            raise Exception("Not able to save: The model was not fitted yet")
        
    