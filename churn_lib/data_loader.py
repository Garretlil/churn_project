import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path

FEATURE_COLS = [
    'age',
    'days_as_customer',
    'total_payment_amount',
    'total_payments_count',
    'avg_payment_amount',
    'days_since_last_activity',
    'total_activity_duration',
    'avg_activity_duration'
]
LABEL_COL='label'

def loadData(filePath):
    return pd.read_parquet(filePath)

def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X=df[FEATURE_COLS].to_numpy()
    y=df[LABEL_COL].to_numpy()
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / (std + 1e-8)
    return [X,y,mean,std]

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test=train_test_split(
        X, y, test_size=test_size, random_state=random_state,stratify=y)
    return [X_train, X_test, y_train, y_test]

def create_tensors(X_train, X_test, y_train, y_test):
    X_train=torch.from_numpy(X_train)
    X_test=torch.from_numpy(X_test)
    y_train=torch.from_numpy(y_train).reshape(-1,1)
    y_test=torch.from_numpy(y_test).reshape(-1,1)
    return [X_train, X_test, y_train, y_test]

def getData():
    base_dir = Path(__file__).parent.parent 
    file_path = base_dir / 'data' / 'features' / 'features.parquet'
    
    df = loadData(str(file_path))
    data=prepare_features(df)
    sd=split_data(data[0],data[1])
    prepared_data=create_tensors(sd[0],sd[1],sd[2],sd[3])
    return prepared_data,data[2],data[3]

