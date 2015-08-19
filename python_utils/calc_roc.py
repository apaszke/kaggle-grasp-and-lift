import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from os import listdir
from os.path import isfile, join
from time import sleep
from sys import platform

val_path = 'sampled_files'
sampled_files = [ f for f in listdir(val_path) if isfile(join(val_path,f)) ]
num_classes = 6

avg_roc_all = 0
plt.figure(1)
for index, file in enumerate(sampled_files):
    print(file)
    path = join(val_path, file)
    subj = int(file.split("_")[0][4:])
    series = int(file.split("_")[1][6:])
    model_df = pd.read_csv(path, header=None)
    events_df = pd.read_csv('data/train/subj{}_series{}_events.csv'.format(subj, series))
    events_df.drop('id', axis=1, inplace=True)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(events_df.iloc[:, i].values, model_df.iloc[:, i].values)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(events_df.values.ravel(), model_df.values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    avg_roc = 0
    for i in range(num_classes):
        avg_roc += roc_auc[i]
    avg_roc /= num_classes
    avg_roc_all += avg_roc

    if index == 1:
        plt.figure(2)
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                                 ''.format(i+1, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Class-wise curves for subject {}, series {}'.format(subj, series))
        plt.legend(loc="lower right")
        plt.figure(1)

    plt.plot(fpr["micro"], tpr["micro"], label='{}, {} (area = {:0.2f})'
                                             ''.format(subj, series, roc_auc["micro"]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average AUC: {:0.6f}'.format(avg_roc_all / len(sampled_files)))
plt.legend(loc="lower right")

# non-blocking show doesn't work on my linux
if platform == 'linux':
    plt.show()
else:
    plt.show(block=False)
    input('Press ENTER to continue... ')
