import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import BinaryAccuracy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from training_model import type_name, batch_size, save_path, device, test_file_path, NB15Dataset, DNN

def load_test():
    print('Loading data ...')
    test = pd.read_csv(test_file_path, low_memory=False) 
    print('testing label destribute:')
    print(test['label'].value_counts())

    test_label_data = test['label']
    test.drop(columns = ['label'], inplace=True)

    test_x, test_y = test, test_label_data

    test_set = NB15Dataset(dfX=test_x, dfY=test_y)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    print("test_set: ", len(test_set))
    print("test_loader: ", len(test_loader))
    return test_y.to_numpy(), test_loader

def calculate_f1(cm, test_y, preds):
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    precision, recall, f1, _ = precision_recall_fscore_support(test_y, preds, average='binary')
    with open(os.path.join(save_path, 'test_acc.txt'), 'a') as f:
        f.write(f"F1 Score : {f1}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision : {precision}\n")
        f.write(f"True Positive Rate (TPR) : {TPR}\n")
        f.write(f"True Negative Rate (TNR) : {TNR}\n")
        f.write('\n')
        
def draw_confusion_matrix(test_y, preds):
    cm=confusion_matrix(test_y, preds)
    calculate_f1(cm, test_y, preds)
    plt.figure(figsize=(8,8))
    plt.title(type_name)
    sns.heatmap(cm,square=True,annot=True,fmt='d',linecolor='white',cmap='Greens',linewidths=1.5,cbar=False)
    plt.xlabel('Pred',fontsize=20)
    plt.ylabel('True',fontsize=20)
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))

def predict(test_loader):
    acc = BinaryAccuracy(threshold = 0.5, device=device)
    tmodel = torch.load(os.path.join(save_path, 'model', 'model')) 
    tmodel.eval()
    test_acc = 0.0
    preds = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = tmodel(inputs)
            preds.append(outputs.detach().cpu())
            test_acc = acc.update(outputs, labels).compute().item()
    print('Acc: {:1.6f}'.format(test_acc))
    preds = torch.cat(preds, dim=0).numpy()
    with open(os.path.join(save_path, 'pred.csv'), 'w') as f:
        f.write("Id,pred\n")
        for i, p in enumerate(preds):
            f.write(f'{i},{p}\n')
    with open(os.path.join(save_path, 'test_acc.txt'), 'a') as f:
        f.write(f"{type_name} : \nAcc : {test_acc}\n")
    return preds

def draw_pred(min = -0.1, mix = 1.1, targets=None, preds=None):
    ''' Plot prediction of your DNN '''
    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.01)
    plt.plot([min, mix], [0.5, 0.5], c='b')
    plt.plot([0.5, 0.5], [min, mix], c='b')
    plt.xlim(min, mix)
    plt.ylim(min, mix)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.savefig(os.path.join(save_path, "pred.png"))
    plt.close('all')

if __name__ == '__main__':    
    test_y, test_loader = load_test()
    preds = predict(test_loader)

    #draw
    draw_confusion_matrix(test_y, np.where(preds < 0.5, 0, 1))
    draw_pred(targets = test_y, preds = preds)