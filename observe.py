import pandas as pd
import matplotlib.pyplot as plt
import os

def _plot(dataframe, savepath, t):
    zero_percentage = (dataframe == 0).sum(axis=1) / dataframe.shape[1]
    zero_percentage = zero_percentage.sort_values()

    # 繪製圖表
    plt.figure(figsize=(10, 5))
    plt.plot(range(dataframe.shape[0]), zero_percentage * 100, color='blue')
    plt.xlabel('Row Index')
    plt.ylabel('Percentage of Zeros')
    plt.title(f'Accumulated percentage of Zeros in Each Row (Label == {t})')
    plt.savefig(os.path.join(savepath, f'label{t}.png'))

def plot_zero_persentage(dataframe, savepath):
    df0 = dataframe[dataframe['label'] == 0]
    _plot(df0, savepath, 0)
    df1 = dataframe[dataframe['label'] == 1]
    _plot(df1, savepath, 1)