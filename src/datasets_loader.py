"""
Dataset loading for Adaptive Self-Retrieval experiments.
Uses the same datasets and splits as TS-RAG and RAFT for direct comparison.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_CONFIGS = {
    'ETTh1': {
        'file': 'ETT-small/ETTh1.csv',
        'target': 'OT',
        'freq': 'h',
        'prediction_lengths': [96, 192, 336, 720],
        'train_end': 8544,
        'val_end': 11425,
    },
    'ETTh2': {
        'file': 'ETT-small/ETTh2.csv',
        'target': 'OT',
        'freq': 'h',
        'prediction_lengths': [96, 192, 336, 720],
        'train_end': 8544,
        'val_end': 11425,
    },
    'ETTm1': {
        'file': 'ETT-small/ETTm1.csv',
        'target': 'OT',
        'freq': '15min',
        'prediction_lengths': [96, 192, 336, 720],
        'train_end': 34176,
        'val_end': 45695,
    },
    'ETTm2': {
        'file': 'ETT-small/ETTm2.csv',
        'target': 'OT',
        'freq': '15min',
        'prediction_lengths': [96, 192, 336, 720],
        'train_end': 34176,
        'val_end': 45695,
    },
    'Weather': {
        'file': 'weather/weather.csv',
        'target': 'OT',
        'freq': '10min',
        'prediction_lengths': [96, 192, 336, 720],
    },
    'ECL': {
        'file': 'electricity/electricity.csv',
        'target': 'OT',
        'freq': 'h',
        'prediction_lengths': [96, 192, 336, 720],
    },
    'Traffic': {
        'file': 'traffic/traffic.csv',
        'target': 'OT',
        'freq': 'h',
        'prediction_lengths': [96, 192, 336, 720],
    },
    'Exchange': {
        'file': 'exchange_rate/exchange_rate.csv',
        'target': 'OT',
        'freq': 'd',
        'prediction_lengths': [96, 192, 336, 720],
    },
    'ILI': {
        'file': 'illness/national_illness.csv',
        'target': 'OT',
        'freq': 'w',
        'prediction_lengths': [24, 36, 48, 60],
    },
}


def load_dataset(name, data_dir='./data'):
    """Load dataset and return train/val/test splits."""
    config = DATASET_CONFIGS[name]
    filepath = os.path.join(data_dir, config['file'])
    df = pd.read_csv(filepath)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    target_col = config['target']
    series = df[target_col].values.astype(np.float32)

    if 'train_end' in config:
        train = series[:config['train_end'] + 1]
        val = series[config['train_end'] + 1:config['val_end'] + 1]
        test = series[config['val_end'] + 1:]
    else:
        n = len(series)
        train = series[:int(0.7 * n)]
        val = series[int(0.7 * n):int(0.8 * n)]
        test = series[int(0.8 * n):]

    return train, val, test, config
