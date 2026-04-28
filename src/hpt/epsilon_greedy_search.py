import numpy as np
import random
from collections import deque
from reservoirpy.nodes import IPReservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli

from sklearn.metrics import r2_score, mean_absolute_percentage_error

class BaseEpsilonGreedyReservoirHPSearch:
    def __init__(self, X_train, y_train, X_test, y_test, n_iterations, epsilon_greedy=None, criterion=mean_absolute_percentage_error, optimize_objective="minimize"):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.memory = deque(maxlen=5)
        self.history = deque(maxlen=n_iterations)
        self.epsilon_greedy = epsilon_greedy
        self.best_params = None
        self.criterion = criterion
        self.optimize_objective=optimize_objective
        self.best_score = +np.inf if self.optimize_objective == "minimize" else -np.inf
        self.search_space = {
            "units": [5, 50, 100, 200],
            "sr": {"min": np.log(1e-2), "max": np.log(1e1)},
            "mu": [0, 1], 
            "input_scaling": {"min": np.log(1e-5), "max": np.log(2e2)},
            "learning_rate": {"min": np.log(1e-5), "max": np.log(1e-2)},
            "connectivity": [0.1, 0.5],
            "activation": ["sigmoid", "tanh"],
            "ridge": {"min": np.log(1e-8), "max": np.log(1e1)},
            "seed": 12345,
            "n_instances": 5,
            "epochs": 100,
            "warmup": 15
        }
    
    def log_uniform_sample(self, param_range):
        """Sample from log-uniform distribution"""
        return np.exp(np.random.uniform(param_range["min"], param_range["max"]))
    
    def perturb_log_param(self, value, param_range, noise_scale=0.1):
        """Perturb parameter in log space"""
        log_value = np.log(value)
        noise = np.random.normal(0, noise_scale)
        new_log_value = log_value + noise
        # Clip to original range
        new_log_value = np.clip(new_log_value, param_range["min"], param_range["max"])
        return np.exp(new_log_value)

    def memory_guided_sample(self):
        """Sample parameters with guidance from historical performance and bias toward smaller units"""
        if len(self.memory) > 0 and random.random() < (1 - self.epsilon_greedy if self.epsilon_greedy is not None else 0.7):  # 70% chance to use memory
            base_config = random.choice(list(self.memory))
            params = {}
            
            for key, value_range in self.search_space.items():
                if key == 'units':
                    # Special handling for units to bias toward smaller values
                    current_units = base_config[key]
                    if random.random() < 0.7:  # 30% chance to change units
                        available_units = self.search_space[key]
                        
                        # Create probability distribution favoring smaller units
                        weights = [1/(i+1) for i in range(len(available_units))]
                        weights = np.array(weights) / sum(weights)
                        
                        # If current units are large, increase probability of choosing smaller ones
                        if current_units >= sum(self.search_space[key]) / len(self.search_space[key]):
                            weights[0:2] *= 2  # Double probability for smaller units
                            weights = weights / sum(weights)
                        
                        params[key] = np.random.choice(available_units, p=weights)
                    else:
                        params[key] = current_units
                        
                elif key == 'activation':
                    params[key] = base_config[key]
                    if random.random() < 0.2:  # 20% chance to change
                        params[key] = random.choice(self.search_space[key])
                        
                elif key in ['mu', 'connectivity']:
                    # Linear scale perturbation for parameters between 0 and 1
                    noise = np.random.normal(0, 0.05)  # 5% noise
                    new_value = base_config[key] + noise
                    new_value = np.clip(new_value, value_range[0], value_range[1])
                    params[key] = new_value
                    
                elif key not in ['epochs', 'warmup', 'n_instances', 'seed']:
                    # Log-scale perturbation
                    params[key] = self.perturb_log_param(base_config[key], value_range)
        else:
            # Random sampling with bias toward smaller units
            available_units = self.search_space["units"]
            weights = [1/(i+1) for i in range(len(available_units))]
            weights = np.array(weights) / sum(weights)
            
            params = {
                "units": np.random.choice(available_units, p=weights),
                "sr": self.log_uniform_sample(self.search_space["sr"]),
                "mu": random.uniform(*self.search_space["mu"]),
                "input_scaling": self.log_uniform_sample(self.search_space["input_scaling"]),
                "learning_rate": self.log_uniform_sample(self.search_space["learning_rate"]),
                "connectivity": random.uniform(*self.search_space["connectivity"]),
                "activation": random.choice(self.search_space["activation"]),
                "ridge": self.log_uniform_sample(self.search_space["ridge"])
            }
        
        params["epochs"] = self.search_space["epochs"]
        params["warmup"] = self.search_space["warmup"]
        params["seed"] = self.search_space["seed"]
        params["n_instances"] = self.search_space["n_instances"]
        
        return params
        
    def search(self, n_iterations=20):
        for i in range(n_iterations):
            params = self.memory_guided_sample()
            print(params)
            score = self.evaluate(params)
            params["score"] = score

            self.history.append(params)
            
            improve_condition = score < self.best_score if self.optimize_objective == "minimize" else score > self.best_score
            if improve_condition:
                self.best_score = score
                self.best_params = params
                self.memory.append(params)
                print(f"New best score: {score:.4f}")
                print("Best params:", {k: f"{v:.2e}" if isinstance(v, float) else v 
                                    for k, v in params.items()})
            
            print(f"Iteration {i+1}/{n_iterations}: Score = {score:.4f}")
            
        return self.best_params, self.best_score
    
    def evaluate(self, params):
        raise NotImplementedError("Subclasses must implement evaluate method")


class EpsilonGreedyReservoirHPSearch_R2(BaseEpsilonGreedyReservoirHPSearch):
    def evaluate(self, params):
        mape_lst = []
        variable_seed = params["seed"]
        
        for _ in range(params["n_instances"]):
            reservoir = IPReservoir(
                units=params['units'],
                sr=params['sr'],
                mu=params['mu'],
                input_scaling=params['input_scaling'],
                learning_rate=params['learning_rate'],
                W=uniform(high=1.0, low=-1.0),
                Win=bernoulli,
                rc_connectivity=params['connectivity'],
                input_connectivity=params['connectivity'],
                activation=params['activation'],
                epochs=params['epochs'],
                seed=variable_seed
            )
            
            readout = Ridge(ridge=params['ridge'])
            
            try:
                train_states = reservoir.run(self.X_train)
                readout = readout.fit(train_states, self.y_train, warmup=params['warmup'])
                
                test_states = reservoir.run(self.X_test)
                predictions = readout.run(test_states)
                
                y_test_flat = np.concatenate(self.y_test)
                predictions_flat = np.concatenate(predictions)
                mape = self.criterion(y_test_flat, predictions_flat)
                mape_lst += [mape]
            
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                return -np.inf

            variable_seed += 1

        return np.mean(mape_lst)
