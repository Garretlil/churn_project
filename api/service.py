import sys
from pathlib import Path
sys.path.append('.')
sys.path.append(str(Path(__file__).parent.parent))
import torch
import pickle
from churn_lib.model_utils import ChurnModel
import numpy as np

base_dir = Path(__file__).parent.parent
file_path = base_dir / 'models'

def load_model():
    weights_path=file_path / 'model.pth'
    model = ChurnModel()

    model.load_state_dict(torch.load(weights_path,weights_only=True))
    model.eval()
    return model
    
def predict(X):
    with torch.no_grad():
        preds=model.forward(X)
        labels=(preds>0.1).float()
    return [preds.tolist(),labels.tolist()]

def prepare(X):
    X=torch.tensor(X).float()
    X=(X-params["mean"])/params["std"]
    return X

def inference(X):
    X=prepare(X)
    res=predict(X)
    return res

model=load_model()

with open(file_path / 'params.pkl', 'rb') as f:
    params = pickle.load(f)
    params['mean'] = params['mean'].astype(np.float32)  
    params['std'] = params['std'].astype(np.float32)   