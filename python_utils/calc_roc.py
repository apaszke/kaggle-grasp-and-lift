import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from os import listdir
from os.path import isfile, join

val_path = 'tmp'
sampled_files = [ f for f in listdir(val_path) if isfile(join(val_path,f)) ]
num_classes = 6

for file in sampled_files:
  print(file)
  path = join(val_path, file)
  subj = int(file.split("_")[0][4:])
  series = int(file.split("_")[1][6:])
  model_df = pd.read_csv(path, header=None)
  events_df = pd.read_csv('data/filtered/subj{}_series{}_events.csv.val'.format(subj, series))
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
  print(roc_auc["micro"])

  plt.figure()
  plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
                                             ''.format(roc_auc["micro"]))
  # for i in range(num_classes):
    # plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                               # ''.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Some extension of Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
plt.show()
