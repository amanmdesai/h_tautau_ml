import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing   
from zipfile import ZipFile

file_name = "../input/higgs-boson/training.zip"
  
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    zip.printdir()
    zip.extractall()




def draw_data_corr(data_original):
    data = data_original.copy()
    enc = LabelEncoder()
    data['Label'] = enc.fit_transform(data['Label'])
    data = data.drop(columns=["EventId"])
    corr = data.corr()
    fig, ax = plt.subplots()
    #fig, ax  = plt.figure(figsize=(25,25))
    plt.figure(figsize=(30,30))
    ax = sns.heatmap(corr, annot=True,square=True,cmap="YlGnBu")
    #plt.savefig("../store/corr.pdf")
    return 0

def draw_sig_vs_bkg(data,signal,background):
    fig  = plt.figure(figsize=(25,25))
    i=0
    for var in signal.columns:
        plt.subplot(6,6,i+1)
        #if data.loc[data[var]==-999]:
        #    continue
        plt.hist(signal[var],color='blue',histtype='step',density=True,bins=40,label="Signal")
        plt.hist(background[var],color='orange',histtype='step',density=True,bins=40,label="Background")
        plt.xlabel(var)
        plt.legend()
        plt.ylabel('Normalized Counts/20')
        if var != 'Label':
            plt.xlim(min(data[var])-1,max(data[var])+1)
        i = i + 1
    fig.tight_layout()
    #plt.savefig("../store/higgs_data.pdf")


def model(data):
    enc = LabelEncoder()
    data['Label'] = enc.fit_transform(data['Label'])
    y = data[['Label']]
    #X = data.drop(columns=["EventId","Label","Weight"])
    #print(X,y) 
    X = data[["DER_deltar_tau_lep","DER_mass_MMC","PRI_jet_subleading_pt","PRI_met_phi"]]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=33,test_size=0.25)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    assert X_train.shape[0] == y_train.shape[0]
    clf = GradientBoostingClassifier(loss='exponential',criterion='squared_error')  #criterion='entropy',max_depth=3,min_samples_leaf=5)
    clf.fit(X_train,y_train.values.ravel())
    print(clf.score(X_train,y_train))
    #y_train_pred = clf.predict(X_train)
    #print(metrics.accuracy_score(y_train, y_train_pred))
    y_pred = clf.predict(X_test)
    print("Metrics for Test Data")
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred,labels=["0","1"]))

def main():
    with open("./training.csv") as higgs_train_data:
        data = pd.read_csv(higgs_train_data)
    #print(data.shape)
    #print(signal.shape)
    #print(background.shape)
    draw_data_corr(data)
    signal = data.loc[data['Label']=='s']
    background = data.loc[data['Label']=='b']
    draw_sig_vs_bkg(data,signal,background)
    model(data)

main()
