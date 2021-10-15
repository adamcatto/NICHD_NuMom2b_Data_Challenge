import pandas as pd
from random import sample
from util.get_data import get_balance_test_set, get_pca_index, get_x_y, label_encoded_data
from util.model import get_classifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from dalex.fairness import resample, reweight, roc_pivot

def trail(**kwargs):
    """ 
    Run training and testing. Save each trail (e.g. 5 fold cross validation). 
    In the predefined folder storage foler.

    Parameters:
        **kwargs:
            path (str): dataset file
            clf (str): classifier to use, supports [RF, SVM, DT, Bagging, LR] 
            output (str): folder to store the data
            pca_feat (Optional[list]): if this parameters is used then generate pca 
                for selected feature 
            pca_comp (Optional[int]): number of pca component to generate if pca is used
            gridsearch (bool): true if using gridsearch
            feature_importance (bool): if generate feature importance (RF supported)
            n_trails (int): number of cross validation trails to run
            n_splits (int): number of splits for cross validation
            test_size (float): percentage of dataset used as test set
            ignore_features (list): features to be ignored in the prediction model
    """

    # generate output directory if it does not exist
    output_dir = kwargs["output"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read the data and preprocess it. 
    data = pd.read_csv(kwargs["path"])

    # preprocessing
    
    data = label_encoded_data(data, kwargs["ignore_features"])
    X, y, studyid = get_x_y(data, "OUTCOME", kwargs["ignore_features"])

    if "pca_feat" in kwargs.keys():
        pca_index = get_pca_index(data, kwargs["pca_feat"], kwargs["ignore_features"])
        kwargs["pca_index"] = pca_index

    
    for trail_id in range(kwargs["n_trails"]):
        # running cross validation
        result = run_cross_val(X, y, studyid, **kwargs)

        f = open(output_dir + 'trail_{}.pkl'.format(trail_id), 'wb')
        pickle.dump(result, f) 
        f.close() 


def run_cross_val(X, y, studyid, **kwargs):
    """
    Return the classification result. 
    
    Parameters:
        X (numpy.array): data 
        y (numpy.array): target 
        clf (sklearn classifier or pipeline):
        **kwargs: refer to trail()

    Return:
        result (list): refer to summary() 
    """

    clf = get_classifier(kwargs["gridsearch"], kwargs["clf"])

    ss = StratifiedShuffleSplit(n_splits=kwargs["n_splits"], 
                                test_size=kwargs["test_size"])
    
    result = []
    for train_index, test_index in ss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        studyid_test = studyid[test_index]

        # standarized the data
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train) 
        X_test = sc.transform(X_test) 

        if "pca_feat" in kwargs.keys():
            pca_index = kwargs["pca_index"]
            pca_comp = X_train[:, pca_index]
            pca = PCA(n_components=kwargs["pca_comp"])
            pca.fit(pca_comp)
            pca_comp = pca.transform(pca_comp)

            X_train = np.delete(X_train, pca_index, 1)
            X_train = np.concatenate([X_train, pca_comp], axis=1)

            pca_comp = pca.transform(X_test[:, pca_index])
            X_test = np.delete(X_test, pca_index, 1)
            X_test = np.concatenate([X_test, pca_comp], axis=1)


        X_test, y_test, studyid_test = get_balance_test_set(X_test, y_test, studyid_test)


        clf.fit(X_train, y_train)
        result.append(summary(clf, X_test, y_test, kwargs["feature_importance"], studyid_test))

    return result


def summary(classifier, X_test, y_test, feature_imp, studyid_test):
    """
    Save the result for a given classifier. 

    Parameters: 
        classifier (sklearn clf): 
        X_test (list):
        y_test (list)
        studyid_test (list): Studyid of the test set. 

    Returns:
        result (list): Returns a list of 
            [feature importance, test data, test target, studyid ]
    

    """
    
    # some model does not have predict_prob therefore we use 
    # a different method to get the prediciton probability
    if "predict_proba" in dir(classifier):
        y_score = classifier.predict_proba(X_test)[:,1]
    else:
        y_score = classifier.decision_function(X_test)

    # save the feature importance
    if feature_imp:
        fi = classifier[1].feature_importances_
        result = [fi, X_test, y_test, y_score, studyid_test]
    else:
        result = [[], X_test, y_test, y_score, studyid_test]

    return result 