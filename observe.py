import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def _plot(dataframe, savepath, t):
    zero_percentage = ((dataframe == 0).sum(axis=1)-1) / dataframe.shape[1]
    zero_percentage = zero_percentage.sort_values().tolist()

    # 將機率分成10個區間
    num_bins = 10
    hist, bins = np.histogram(zero_percentage, bins=num_bins, range=(0,1))

    # 區間中心
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 繪製圖表
    plt.figure(figsize=(10, 10))
    plt.bar(bin_centers, hist, width=(bins[1] - bins[0])*0.8, color='blue', alpha=0.7)
    plt.xlabel('Zero percentage')
    plt.ylabel('Numbers')
    plt.title(f'Histogram of data zero_percentage (Label is {t})')
    plt.savefig(os.path.join(savepath, f'label{t}.png'))

def plot_zero_persentage(dataframe, savepath):
    df0 = dataframe[dataframe['label'] == 0]
    _plot(df0, savepath, 0)
    df1 = dataframe[dataframe['label'] == 1]
    _plot(df1, savepath, 1)