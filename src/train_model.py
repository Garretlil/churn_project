import sys
from pathlib import Path
sys.path.append('.')
sys.path.append(str(Path(__file__).parent.parent))

import pickle

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from churn_lib.data_loader import getData
from churn_lib.model_utils import ChurnModel
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


EPOCHS=200
    
data,mean,std=getData()
dataset=TensorDataset(data[0],data[2])
train_loader=DataLoader(dataset,batch_size=32)

model=ChurnModel()

criterion = nn.BCELoss()
optimizer=torch.optim.Adam(lr=0.0001,params=model.parameters())


for epoch in range(EPOCHS):
    epoch_loss=0.0

    for batch_x,batch_y in train_loader:
        batch_x = batch_x.float()
        batch_y = batch_y.float().reshape(-1,1)
        optimizer.zero_grad()
        pred=model.forward(batch_x)

        loss=criterion(pred,batch_y)

        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

    if epoch%10==0:
        print(epoch_loss/len(train_loader))    


def evaluate_model(model,dataset):
    model.eval()
    probs=[]
    labels=[]
    with torch.no_grad():
        for batch_x,batch_y in dataset:
            batch_x=batch_x.float()
            batch_y=batch_y.float().reshape(-1,1)
            pred=model.forward(batch_x)
            probs.extend(pred.numpy())
            labels.extend((pred>0.5).float().numpy())
    return [probs,labels]


dataset=TensorDataset(data[1],data[3])
test_loader=DataLoader(dataset,batch_size=32)


res=evaluate_model(model,dataset)
probs=np.array(res[0]).flatten()
labels=np.array(res[1]).flatten()
true_labels=dataset.tensors[1].numpy().flatten()

metrics={
    "Accuracy":0,
    "AUC":0,
    "Precision":0,
    "Recall":0,
    "F1-score":0
}
      
metrics["Accuracy"]=int((res[1]==true_labels).sum()/len(probs))
metrics["Precision"]=int(precision_score(true_labels,labels))
metrics["AUC"]=roc_auc_score(true_labels,probs)
metrics["Recall"]=recall_score(true_labels,labels)
metrics["F1-score"]=f1_score(true_labels,labels)

base_dir = Path(__file__).parent.parent
file_path = base_dir / 'models'
torch.save(model.state_dict(),f'{file_path}/model.pth')

norm_params={
    "mean":mean,
    "std":std
}

with open(f'{file_path}/params.pkl','wb') as f:
    pickle.dump(norm_params,f)

print(metrics)