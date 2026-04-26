import numpy as np
from reservoirpy.nodes import IPReservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import random
from collections import deque
import os
import pandas as pd

from scipy.stats import loguniform
from scipy.stats import spearmanr

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

class ReservoirVisualizer:
    def __init__(self, save_dir="./reservoir_plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def create_history_df(self, searcher):
        """Create DataFrame from search history"""
        data = []
        for i, entry in enumerate(searcher.history):
            print("entry", entry)
            params_copy = entry.copy()  # Extract just the params, not the full entry
            params_copy['iteration'] = i
            params_copy['score'] = entry["score"]  # Use the stored score instead of calling `evaluate`
            data.append(params_copy)
        return pd.DataFrame(data)


    def plot_optimization_history(self, searcher, df):
        """Plot optimization history similar to Optuna"""
        fig = go.Figure()
        
        # Add all trials
        fig.add_trace(go.Scatter(
            x=df['iteration'],
            y=df['score'],
            mode='markers',
            name='Trials',
            marker=dict(color='lightblue')
        ))
        
        # Add best score
        best_scores = df['score'].cummax()
        fig.add_trace(go.Scatter(
            x=df['iteration'],
            y=best_scores,
            mode='lines',
            name='Best Score',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Optimization History',
            xaxis_title='Iteration',
            yaxis_title='Score',
            template='plotly_white'
        )
        
        fig.write_image(f"{self.save_dir}/optimization_history.pdf")
        return fig

    def plot_param_importances(self, df):
        """Plot parameter importances based on Spearman correlation"""
        importances = {}
        for column in df.columns:
            if column not in ['iteration', 'score', 'epochs', 'warmup', 'activation']:
                correlation, _ = spearmanr(df[column], df['score'])
                importances[column] = abs(correlation)
        
        # Sort importances
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(importances.keys()),
            y=list(importances.values())
        ))
        
        fig.update_layout(
            title='Parameter Importances',
            xaxis_title='Parameter',
            yaxis_title='Importance Score',
            template='plotly_white'
        )
        
        fig.write_image(f"{self.save_dir}/param_importances.pdf")
        return fig

    def plot_parallel_coordinates(self, df):
        """Plot parallel coordinates"""
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=df['score'],
                         colorscale='Viridis'),
                dimensions=[
                    dict(range=[df[col].min(), df[col].max()],
                         label=col,
                         values=df[col])
                    for col in df.columns
                    if col not in ['activation', 'epochs', 'warmup']
                ]
            )
        )
        
        fig.update_layout(
            title='Parallel Coordinates Plot',
            template='plotly_white'
        )
        
        fig.write_image(f"{self.save_dir}/parallel_coordinates.pdf")
        return fig

    def plot_contour(self, df, param1, param2):
        """Plot contour for two parameters"""
        fig = go.Figure(data=
            go.Contour(
                x=df[param1],
                y=df[param2],
                z=df['score'],
                colorscale='Viridis'
            )
        )
        
        fig.update_layout(
            title=f'Contour Plot: {param1} vs {param2}',
            xaxis_title=param1,
            yaxis_title=param2,
            template='plotly_white'
        )
        
        fig.write_image(f"{self.save_dir}/contour_{param1}_{param2}.pdf")
        return fig

    def plot_slice(self, df):
        """Plot slice plots for each parameter"""
        figs = []
        for param in df.columns:
            if param not in ['iteration', 'score', 'epochs', 'warmup', 'activation']:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df[param],
                    y=df['score'],
                    mode='markers',
                    marker=dict(color='blue', size=8)
                ))
                
                fig.update_layout(
                    title=f'Slice Plot: {param}',
                    xaxis_title=param,
                    yaxis_title='Score',
                    template='plotly_white'
                )
                
                fig.write_image(f"{self.save_dir}/slice_{param}.pdf")
                figs.append(fig)
        
        return figs

def visualize_search(searcher, save_path):
    """Generate all visualizations"""
    visualizer = ReservoirVisualizer(save_path)
    df = visualizer.create_history_df(searcher)
    
    # Generate all plots
    visualizer.plot_optimization_history(searcher, df)
    visualizer.plot_param_importances(df)
    visualizer.plot_parallel_coordinates(df)
    
    # Generate contour plots for some important parameter pairs
    important_params = ['learning_rate', 'ridge', 'sr', 'input_scaling']
    for i in range(len(important_params)):
        for j in range(i+1, len(important_params)):
            visualizer.plot_contour(df, important_params[i], important_params[j])
    
    visualizer.plot_slice(df)
