import pickle
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from dalex.fairness._group_fairness.plot import plot_fairness_check
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def fairness_check(subgroup_df, visit, classifier, bias_feature, privileged, optional_threshold, remove_subgroup):
    
    directory = "./result/visit{}/{}/".format(visit, classifier)

    
    metrics = ["eop", "ppr", "per", "aer", "spr"]
    label = "XGB"
    
    subgroup_list, subgroup_confusion_matrix = merge_result(subgroup_df, directory, bias_feature, 0.5, optional_threshold)
    fairness_data = []

    tn_w, fp_w, fn_w, tp_w = 0, 0, 0, 0
    for protected in subgroup_list:
        tn, fp, fn, tp = subgroup_confusion_matrix[protected]

        tn_w += tn
        fp_w += fp
        fn_w += fn
        tp_w += tp

        


        if protected == privileged:
            continue
        for metric in metrics:
            score = get_ratio(protected, privileged, subgroup_confusion_matrix, metric)
            fairness_data.append([protected, metric, score - 1, label])

    fairness_data  = pd.DataFrame(fairness_data)
    
    fairness_data.columns = ["subgroup", "metric", "score", "label"]
    fairness_data = fairness_data[fairness_data["subgroup"] != remove_subgroup]
    fairness_data.loc[fairness_data.metric == 'eop', 'metric'] = 'Equal opportunity ratio     TP/(TP + FN)'
    fairness_data.loc[fairness_data.metric == 'aer', 'metric'] = 'Accuracy equality ratio    (TP + TN)/(TP + FP + TN + FN)'
    fairness_data.loc[fairness_data.metric == 'ppr', 'metric'] = 'Predictive parity ratio     TP/(TP + FP)'
    fairness_data.loc[fairness_data.metric == 'per', 'metric'] = 'Predictive equality ratio   FP/(FP + TN)'
    fairness_data.loc[fairness_data.metric == 'spr', 'metric'] = 'Statistical parity ratio   (TP + FP)/(TP + FP + TN + FN)'


    fig = plot_fairness_check(fairness_data, 1, 0.8, "Fairness Check")
    fig.show()



def parity_loss(subgroup_df, visit, classifier, bias_feature, privileged, threshold, optional_threshold, remove_subgroup):
    
    directory = "./result/visit{}/{}/".format(visit, classifier)

    subgroup_list, subgroup_confusion_matrix = merge_result(subgroup_df, directory, bias_feature, threshold, optional_threshold)

    metrics = ["eop", "ppr", "per", "aer", "spr"]

    parity_data = {}
    
    for metric in metrics:
        parity_data[metric] = 0
        for protected in subgroup_list:
            if protected == privileged or protected == remove_subgroup:
                continue
            score = get_ratio(protected, privileged, subgroup_confusion_matrix, metric)
            parity_data[metric] += abs(np.log(score))
    return parity_data


def get_confusion_matrix(subgroup_list, subgroup_map, y_test, y_pred):
    subgroup_confusion_matrix = {}
    for group in subgroup_list:
        index = np.where(subgroup_map == group)
        subgroup_confusion_matrix[group] = confusion_matrix(y_test[index], y_pred[index]).ravel()
    return subgroup_confusion_matrix



def get_ratio(protected, privileged, subgroup_confusion_matrix, metric):
    tn, fp, fn, tp = subgroup_confusion_matrix[protected]
    priv_tn, priv_fp, priv_fn, priv_tp = subgroup_confusion_matrix[privileged]

    if metric == "eop":
        protected_tpr = tp / (tp + fn)
        priv_tpr = priv_tp / (priv_tp + priv_fn)
        return protected_tpr/ priv_tpr

    elif metric == "ppr":
        protected_ppv = tp / (tp + fp)
        priv_ppv = priv_tp / (priv_tp + priv_fp)
        return protected_ppv/ priv_ppv
    
    elif metric == "per":
        protected_fpr = fp / (fp + tn)
        priv_fpr = priv_fp / (priv_fp + priv_tn)
        return protected_fpr/ priv_fpr
    
    elif metric == "aer":
        protected_acc = (tp + tn) / (fp + tp + tn +fn)
        priv_acc = (priv_tp + priv_tn) / (priv_fp + priv_tp + priv_tn + priv_fn)
        return protected_acc / priv_acc

    elif metric == "spr":
        protected_stp = (tp + fp) / (fp + tp + tn +fn)
        priv_stp = (priv_tp + priv_fp) / (priv_fp + priv_tp + priv_tn + priv_fn)
        return protected_stp / priv_stp


def merge_result(subgroup_df, directory, bias_feature, threshold, optional_threshold):
    studyid = []
    y_test = []
    y_score = []

    for file in os.listdir(directory):
        f = open(directory + file, "rb")

        trail = pickle.load(f)
        for result in trail:
            studyid.extend(result[4])
            y_test.extend(result[2])
            y_score.extend(result[3])

    studyid = np.array(studyid)
    y_test = np.array(y_test)
    y_score = np.array(y_score)

    subgroup_map = {}
    for record in subgroup_df.to_dict("records"):
        subgroup_map[record["PublicID"]] = record[bias_feature]

    subgroup_map = pd.Series(studyid).map(subgroup_map)
    mask = (~subgroup_map.isnull()).values

    y_score, y_test, subgroup_map = y_score[mask], y_test[mask], subgroup_map[mask]
    y_pred = np.array([1 if x >= threshold else 0 for x in y_score])

    for subgroup, value in optional_threshold.items():
        subgroup_score = y_score[subgroup_map == subgroup]
        subgroup_score = np.array([1 if x >= value else 0 for x in subgroup_score])

        y_pred[subgroup_map == subgroup] = subgroup_score
        
    
    subgroup_list = np.unique(subgroup_map)
    subgroup_confusion_matrix = get_confusion_matrix(subgroup_list, subgroup_map, y_test, y_pred)

    return subgroup_list, subgroup_confusion_matrix 




def get_fairness_graphs(visit, classifier, optional_threshold):
    subgroup_df = pd.read_csv("./data/V{}.csv".format(visit))

    race_map = {1: "White",
                2: "Black",
                3: "Hispanic",
                4: "Other", 
                5: "Asian", 
                6: "Other",
                7: "Other",
                8: "Multiracial", 
                9: "Other"}

    subgroup_df["Race"] = subgroup_df.Race.map(race_map)
    fairness_check(subgroup_df, visit, classifier, "Race", "White", optional_threshold, "Other")


def plot_parity_loss(visit, classifier, race):
    subgroup_df = pd.read_csv("./data/V{}.csv".format(visit))

    race_map = {1: "White",
                2: "Black",
                3: "Hispanic",
                4: "Other", 
                5: "Asian", 
                6: "Other",
                7: "Other",
                8: "Multiracial", 
                9: "Other"}


    subgroup_df["Race"] = subgroup_df.Race.map(race_map)

    metric_map = {'eop': "TPR", "ppr": "PPV", "per": "FPR", "aer": "ACC", "spr": "STP"}

    threshold = 0.0

    parity_data = {}
    parity_x = {}
    metrics = ["eop", "ppr", "per", "aer", "spr"]
    for metric in metrics:
        parity_data[metric] = []
        parity_x[metric] = []

    for x in range(100):
        threshold += 0.01
        pl = parity_loss(subgroup_df, visit, classifier, "Race", "White", 0.5, { race:threshold}, remove_subgroup="Other")

        for metric in metrics:
            if pl[metric] < 4:
                parity_x[metric].append(threshold)
                parity_data[metric].append(pl[metric])
    
    for metric in metrics:
        x = parity_x[metric]
        y = parity_data[metric]
        x = x[:len(y)]

    
        
        
        plt.plot(x, y, label=metric_map[metric])
    
    plt.vlines(0.5, 0, 4, linestyles= "--", label= "Original Threshold", color="gray", alpha=0.3)
    plt.vlines(0.52, 0, 4, linestyles= "--", label= "Optimal Threshold", color="r", alpha=0.3)
    plt.text(x=0.55, y=3.8, s='minimum: {}'.format(0.52), alpha=0.7, color='#334f8d')

    plt.xlabel("Threshold for {} Race, Other Race Threshold Constant".format(race))
    plt.ylabel("Parity Loss")
    plt.title("{} Model".format(classifier))
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(12.5, 8.5)
    plt.show()


if __name__ == "__main__":

    visit = 2
    model  = "xgb"

    get_fairness_graphs(visit, model, {})

    # run parity loss plot
    plot_parity_loss(visit, model, "Black")

    get_fairness_graphs(visit, model, {"Black":0.52})


