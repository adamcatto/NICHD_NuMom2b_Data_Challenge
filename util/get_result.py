from sklearn.metrics import roc_auc_score
import pickle
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from dalex.fairness._group_fairness.plot import plot_fairness_check
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def auc_summary(results_dir, output_dir, name):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    auc = []
    y_true = []
    y_pred = []
    for file in os.listdir(results_dir):
        f = open(results_dir + file, "rb")

        trail = pickle.load(f)
        for result in trail:
            auc.append(roc_auc_score(result[2], result[3]))
            a = list(result[2])
            b = list((result[3] >= 0.5).astype(int))
            y_true.extend(a)
            y_pred.extend(b)



    auc = np.array(auc)
    auc = pd.DataFrame(auc)
    auc.columns = [name]
    
    auc.to_csv(output_dir + name, index=False)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()


    print(len(y_pred), sum(y_pred))
    print("AUC: {} pm {}".format(auc.mean(), auc.std()))
    print("Sensitivity (TP/ TP+FN): {}".format(TP/ (TP+FN)))
    print("Specificity (TN/ FP + TN): {}".format(TN/ (FP + TN)))
    print("PPV (TP/ TP +FP): {}".format(TP/ (TP +FP)))
    print("NPV (TN/FN+TN): {}".format(TN/ (FN+TN)))


def feature_imp_summary(directory, output_dir, features, name):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fi = []
    for file in os.listdir(directory):

        f = open(directory + file, "rb")

        trail = pickle.load(f)
        for result in trail:
            fi.append(result[0])


    fi = np.array(fi).mean(axis=0)

    fi = pd.DataFrame([features, fi]).T
    fi.columns = ["Feature", "Feature Importance"]

    fi.to_csv(output_dir + name, index=False)


def get_feature_names(path, pca_feat = None, n_comp = 10):
    data = pd.read_csv(path)
    ignore_columns = ["PublicID", "OUTCOME"]
    data = data.drop(ignore_columns, axis =1) 
    
    if pca_feat:
        data = data.drop(pca_feat, axis =1) 

    features = list(data.columns)

    if pca_feat:
        features.extend(["PC" + str(x+1) for x in range(10)])

    return features

