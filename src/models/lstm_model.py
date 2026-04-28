import time
import os
import numpy as np
import optuna

from src.hpt.hp_visualizer import visualize_search
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from utils.io import read_json_file, write_json_file, write_torch_model, load_torch_model
from utils.logger import get_logger
from utils.custom_metrics import mape_loss

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
logger = get_logger("LSTMRegressorModel")

class LSTMRegressorModel(nn.Module):
    def __init__(self, name: str, version=None, params: dict = None, criterion = mape_loss, optimize_objective="minimize"):
        super().__init__()
        self.name = name
        self.version = version or str(int(time.time()))
        self.params = params or {}
        self.hidden_size = params.get("hidden_size", 64)
        self.num_layers = params.get("num_layers", 2)
        self.device = params.get("device", "cpu")
        self.epochs = params.get("epochs", 20)
        self.patience = params.get("patience", 5)
        self.lstm = nn.LSTM(
            input_size=params.get("input_size", 1), 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=params.get("dropout", 0)
        )
        self.fc = nn.Linear(self.hidden_size, 1)
        self.optimize_objective = optimize_objective
        self.to(self.device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.get("lr", 1e-4))
        self.history = {"train_loss": [], "val_loss": []}

    def forward(self, X):
        X = X.to(self.device)
        out, (hn, cn) = self.lstm(X) 
        return self.fc(hn[-1])
    
    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                preds = self(batch_X)
                loss = self.criterion(preds, batch_y)
                val_loss += loss.item()
        return val_loss / len(val_loader)
    
    def fit(self, train_dataloader, val_dataloader=None):
        start_time = time.time()
        best_val_loss = float('inf') if self.optimize_objective == "minimize" else float('-inf')
        
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                batch_y = batch_y.to(self.device)
                predictions = self(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
            
                train_loss += loss.item()
        
            avg_train_loss = train_loss / len(train_dataloader)
            self.history["train_loss"].append(avg_train_loss)

            if val_dataloader is not None:
                avg_val_loss = self.validate(val_dataloader)
                self.history["val_loss"].append(avg_val_loss)
                
                logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

                improve_condition = avg_val_loss < best_val_loss if self.optimize_objective == "minimize" else avg_val_loss > best_val_loss
                if improve_condition:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    self.best_state = self.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        if hasattr(self, 'best_state'):
                            self.load_state_dict(self.best_state)
                        break
            else:
                logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")

        self.training_time = time.time() - start_time
        logger.info(f"Training time: {self.training_time}") 

    def predict(self, dataloader):
        self.eval()
        all_preds = []
        all_actuals = []

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                preds = self(batch_X)
                all_preds.append(preds.detach().cpu().numpy())
                all_actuals.append(batch_y.detach().cpu().numpy())

        return np.concatenate(all_actuals).flatten(), np.concatenate(all_preds).flatten()
    
    @staticmethod
    def tune(train_loader, val_loader, name, n_trials=20, device="cpu"):
        start_time = time.time()
        def objective(trial):
            params = {
                "hidden_size": trial.suggest_int("hidden_size", 16, 128),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "epochs": 20,
                "input_size": next(iter(train_loader))[0].shape[2],
                "device": device
            }
            
            model = LSTMRegressorModel(name=name, params=params)
            model.fit(train_loader)
            y_true, y_pred = model.predict(val_loader)
            return mean_absolute_percentage_error(y_true, y_pred)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter search time: {time.time() - start_time}")
        logger.info(f"Best params: {study.best_params}")
        
        best_params = study.best_params
        best_params["input_size"] = next(iter(train_loader))[0].shape[2]
        best_params["device"] = device

        hpt_model = LSTMRegressorModel(name=name, params=best_params)
        hpt_model.fit(train_loader)
        hpt_model.training_time = time.time() - start_time

        return hpt_model
    
    def evaluate(self, y_true, y_pred, eval_function="mape"):
        if eval_function == "mape":
            return mean_absolute_percentage_error(y_true, y_pred)
        elif eval_function == "r2":
            return r2_score(y_true, y_pred)
        else:
            raise Exception(f"Error: The evaluation function {eval_function} is not suported")


    @staticmethod
    def load_model(name, version, base_path: str = "./data/models"):
        model_path = os.path.join(base_path, name, version)
        params = read_json_file(os.path.join(model_path, "params.json"))

        instance = LSTMRegressorModel(name=name, version=version, params=params)
        load_torch_model(instance, os.path.join(model_path, "model.pth"), device=params.get("device", "cpu"))
        return instance
    

    def save_model(self, base_path: str = "./data/models"):
        logger.info(f"The model params are {self.params}")
        model_path = os.path.join(base_path, self.name, self.version)
        write_json_file(self.params, os.path.join(model_path, "params.json"))
        write_torch_model(self, os.path.join(model_path, "model.pth"))

    def plot_history(self, title: str = "Train history"):
        try:
            train_loss = self.history["train_loss"]
            val_loss = self.history["val_loss"]
            if len(train_loss) != len(val_loss):
                print(f"Error: Array shapes do not match ({train_loss.shape} vs {val_loss.shape}).")
                return

            plt.figure(figsize=(10, 6))
            plt.plot(train_loss, linestyle='-', label='train loss')
            plt.plot(val_loss, linestyle='--', label='val loss')
            
            plt.title(title)
            plt.xlabel("epochs")
            plt.ylabel("MAPE")
            plt.legend()
            
            plt.show()
            print("Plot rendered successfully.")
            
        except Exception as e:
            print(f"An error occurred while plotting: {e}")
        
    