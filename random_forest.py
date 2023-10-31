import os, sys, time

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class Logger(object):
    def __init__(self, filename='Log/default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    @classmethod
    def timestamped_print(self, *args, **kwargs):
        _print(time.strftime("[%Y/%m/%d %X]"), *args, **kwargs)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def log_history(name_s_log):
    # log
    createFolder('Log/')
    sys.stdout = Logger('Log/' + name_s_log + '.log', sys.stdout)
    sys.stderr = Logger('Log/' + name_s_log + '.err', sys.stderr)

if __name__ == '__main__':
    _print = print
    print = Logger.timestamped_print
    log_history(os.path.basename(__file__))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import sys, os

def RandomForest(train_x, test_x, train_y, test_y):
    # 顯示資料集資訊
    print('train data detail:')
    print(train_x.shape)
    print(train_y.value_counts().to_dict())
    print('test data detail:')
    print(test_x.shape)
    print(test_y.value_counts().to_dict())

    # 建立隨機森林分類器
    rf = RandomForestClassifier(random_state=42)

    # 訓練模型
    rf.fit(train_x, train_y)

    # 進行預測
    predictions = rf.predict(test_x)
    accuracy = (predictions == test_y).mean()
    print(f'Accuracy: {accuracy}\n')
    return rf

if __name__ == '__main__':
    # 讀檔-training
    filename = os.path.join('data', '4_sum', 'train_data.csv')
    train_x = pd.read_csv(filename, low_memory=False)
    del filename

    train_y = train_x['label']
    train_y.replace('benign', 0, inplace=True)
    train_y.replace('malicious', 1, inplace=True)

    train_x.drop(columns=['label', 'sum'], inplace=True)

    # -------------------------------------------------------------------

    # Test data with no mapping
    print('Test data with no mapping')
    # 讀檔-test
    filename = os.path.join('data','0_ori', 'test.csv')
    test_x = pd.read_csv(filename, low_memory=False)
    test_x = test_x.drop_duplicates()
    del filename

    test_y = test_x['label']

    test_x.drop(columns=['label'], inplace=True)

    RandomForest(train_x, test_x, train_y, test_y)
    del test_x, test_y

    # -------------------------------------------------------------------

    # Test data with mapping - benign_test
    print('Test data with mapping - benign_test')
    # 讀檔-test
    filename = os.path.join('data','6_mapped_test', 'benign_test.csv')
    test_x = pd.read_csv(filename, low_memory=False)
    test_x = test_x.drop_duplicates()
    del filename

    test_y = test_x['label']

    test_x.drop(columns=['label'], inplace=True)

    RandomForest(train_x, test_x, train_y, test_y)
    del test_x, test_y
    
    # -------------------------------------------------------------------

    # Test data with mapping - malicious_test
    print('Test data with mapping - malicious_test')
    # 讀檔-test
    filename = os.path.join('data','6_mapped_test', 'malicious_test.csv')
    test_x = pd.read_csv(filename, low_memory=False)
    test_x = test_x.drop_duplicates()
    del filename

    test_y = test_x['label']

    test_x.drop(columns=['label'], inplace=True)
    
    RandomForest(train_x, test_x, train_y, test_y)
    del test_x, test_y

    # -------------------------------------------------------------------

    # Test data with mapping - HS_test
    print('Test data with mapping - HS_test')

    # 讀檔-test
    filename = os.path.join('data','6_mapped_test', 'HS_test.csv')
    test_x = pd.read_csv(filename, low_memory=False)
    test_x = test_x.drop_duplicates()
    del filename

    test_y = test_x['label']

    test_x.drop(columns=['label'], inplace=True)

    RandomForest(train_x, test_x, train_y, test_y)
    del test_x, test_y