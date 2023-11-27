import torch, os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torcheval.metrics import BinaryAccuracy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

# 關閉隨機性
seed = 4222
shuffle = True
torch.manual_seed(seed)
np.random.seed(seed)

class NB15Dataset(Dataset):
    def __init__(self, dfX, dfY):
        self.data = torch.FloatTensor(dfX.values)
        self.label = torch.FloatTensor(np.squeeze(dfY.values))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class DNN(nn.Module):
    def __init__(self, input_dim=73):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x

def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
        print(device)
    elif torch.cuda.is_available():
        device = "cuda"
        print(torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print(device)
    return device

# 設定參數
type_name = 'ori'
train_file_path = os.path.join('data', '1_preprocess', 'train.csv')
test_file_path = os.path.join('data', '1_preprocess', 'test.csv')
# type_name = 'ct'
# train_file_path = os.path.join('data', '3_DataWithPvalue', 'ct-value', 'train.csv')
# test_file_path = os.path.join('data', '6_mapped_test', 'benign_test.csv')
device = get_device()
model = DNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.RAdam(model.parameters(), lr=0.00001)
batch_size = 256
num_epoch = 1000
save_path = os.path.join('result', type_name+'_'+str(num_epoch))

def train_model(train_loader, val_loader):
    print("Starting Training...")
    best_epoch = 0
    best_acc = 0.0
    loss_record = {'train': [], 'dev': []}
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        acc = BinaryAccuracy(threshold = 0.5, device=device)
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward() 
            optimizer.step()
            train_loss += batch_loss.item()
            train_acc = acc.update(outputs, labels).compute().item()
            loss_record['train'].append(batch_loss.item())
        train_loss = train_loss/len(train_loader)

        # validation
        acc = BinaryAccuracy(threshold = 0.5, device=device)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                val_loss += batch_loss.item()
                val_acc = acc.update(outputs, labels).compute().item()
        val_loss = val_loss/len(val_loader)
        loss_record['dev'].append(val_loss)

        print('[{}/{}] Train Acc:{:1.6f} Loss:{:1.6f} | Val Acc:{:1.6f} Loss:{:1.6f}'.format(
                epoch+1, num_epoch, train_acc, train_loss, val_acc, val_loss))
        
        if val_acc >= best_acc:
            best_epoch = epoch + 1
            best_acc = val_acc
            print("Model saving at Epoch: {}".format(best_epoch))
            torch.save(model, os.path.join(save_path, 'model', 'model'))
        else:
            print("Last best model at Epoch: {}".format(best_epoch))

        with open(os.path.join(save_path, 'train.txt'), 'w') as f:
            f.write('Epoch: {}/{} Acc={:1.6f} Loss={:1.6f} Best Epoch={} Best Acc={:1.6f}\n'.format(epoch+1, num_epoch, train_acc, train_loss, best_epoch, best_acc)) 
        with open(os.path.join(save_path, 'train_acc.txt'), 'a') as f:
            f.write(f"{train_acc}\n")
        with open(os.path.join(save_path, 'train_loss.txt'), 'a') as f:
            f.write(f"{train_loss}\n")
        with open(os.path.join(save_path, 'valid_acc.txt'), 'a') as f:
            f.write(f"{val_acc}\n")
        with open(os.path.join(save_path, 'valid_loss.txt'), 'a') as f:
            f.write(f"{val_loss}\n")
    return loss_record

def DNN_preprocess():
    print('Loading data ...')
    train = pd.read_csv(train_file_path, low_memory=False)
    print('training label destribute:')
    print(train['label'].value_counts())

    # 與label分開
    train_label = train['label']
    train.drop(columns = ['label'], inplace=True)

    # 8:2分train與validation
    train_x, val_x, train_y, val_y = train_test_split(train, train_label, test_size=0.2, random_state=seed)

    train_set = NB15Dataset(dfX=train_x, dfY=train_y)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
    print("train_set: ", len(train_set))
    print("train_loader: ", len(train_loader))
    val_set = NB15Dataset(dfX=val_x, dfY=val_y)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=shuffle)
    print("validate_set: ", len(val_set))
    print("validate_loader: ", len(val_loader))

    return train_model(train_loader, val_loader)

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
    plt.xlabel('predicted value',fontsize=20)
    plt.ylabel('ground truth value',fontsize=20)
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close('all')

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

def draw_loss():
    '''Plot learning curve of your DNN (train & dev loss)'''
    total_steps = len(loss_record['train'])
    x1 = range(total_steps)
    x2 = x1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(18, 12))
    plt.plot(x1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Training steps')
    plt.ylabel('BCE loss')
    plt.title('Learning curve of {}'.format(type_name))
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close('all')

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
    # 刪除舊資料夾

    # 創建資料夾
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)

    # train
    loss_record = DNN_preprocess()

    draw_loss()

    # test
    test_y, test_loader = load_test()
    preds = predict(test_loader)

    #draw
    draw_confusion_matrix(test_y, np.where(preds < 0.5, 0, 1))
    draw_pred(targets = test_y, preds = preds)