# script to load the higgs data-set ; relative file paths are used


from turtle import color
import pandas as pd

import matplotlib.pyplot as plt


with open("../dataset/training.csv") as higgs_train_data:
    data = pd.read_csv(higgs_train_data)


signal = data.loc[data['Label']=='s']

background = data.loc[data['Label']=='b']

#print(data.shape)
#print(signal.shape)
#print(background.shape)


def draw_sig_vs_bkg(signal,background):
    fig  = plt.figure(figsize=(20,20))
    i=0
    for var in signal.columns:
        plt.subplot(6,6,i+1)
        if data.loc[data[var]==-999]:
            continue
        plt.hist(signal[var],color='blue',histtype='step',density=True,bins=20)
        plt.hist(background[var],color='orange',histtype='step',density=True,bins=20)
        i = i + 1
    plt.show()

draw_sig_vs_bkg(signal,background)

