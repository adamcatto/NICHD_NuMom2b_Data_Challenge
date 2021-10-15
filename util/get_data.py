from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from random import sample

def label_encoded_data(data, ignore_features):
    """
    Label encodes the categorial features. This is determined by the columns 
    type as np.object in a pandas dataframe. 
    eg. ["yellow", "red", "black", "white", "red"] -> [0, 1, 2, 3, 1]

    Paramters:
        data (DataFrame): dataset 
        ignore_features (list): list feature that we are not interested 
            label encoded. 

    Returns:
        data (DataFrame): dataset that contain label encoded features 
    """


    data_dict = dict(data.dtypes)
    data = data.fillna(0)

    features = list(data.columns)

    for feature in ignore_features:
        features.remove(feature)


    le = LabelEncoder()

    for labels in features:
        # check if the column is categorical 

        if data_dict[labels] == np.object:
            data.loc[:, labels] = data.loc[:, labels].astype(str)
            data.loc[:, labels] = le.fit_transform(data.loc[:, labels])

    return data


def get_pca_index(data, feature_names, ignore_features):
    """ 
    Get the index of the feature intested. We need ignore_feature to account
    for the feature that are removed.

    Paramter:
        data (DataFrame): raw dataset
        feature_name (list): feature that we want to return the index on
        ignore_features (list): features that we want to exclude
        
    Returns:
        feature_index (list): index of interest features
    
    """

    # function to get diet index to use for pca
    X = data.loc[:, ~data.columns.isin(ignore_features)]
    feature_index = [x for x in range(X.shape[1]) if X.columns.isin(feature_names)[x]]
    return feature_index


def get_x_y(data, target, ignore_features):
    """
    Get the target values. Currently support gestational week and PTB vs NPTB.

    Parameter:
        data (DataFrame): raw dataset
        target (str): GAWKSEND or OUTCOME
        ignore_features (list): features that we want to exclude

    Returns:
        X (numpy matrix): feature data
        y (list): target data
        studyid (list): studyid of patients        
    
    """

    X = data.loc[:, ~data.columns.isin(ignore_features)]
    X = X.values
    studyid = data.PublicID.values
    
    y = data.loc[:, "OUTCOME"].values
    return X, y, studyid


def get_balance_test_set(X_test, y_test, studyid_test):
    """
    Balanced the number of postive vs negative example in the test set by 
    randomly removing sample in the majority set. 

    Parameters:
        X_test (numpy matrix): test data in sepcific train test split
        y_test (list): target data in sepcific train test split 
        studyid (list): studyid in sepciic train test split
    
    Returns:
        X_test, y_test, studyid that are balanaced in positive and negative 
        samples 
    """

    pos = [x for x in range(len(y_test)) if y_test[x] == 1 ]
    neg = sample([x for x in range(len(y_test)) if y_test[x] == 0 ], len(pos))
    pos.extend(neg)

    X_test = X_test[pos]
    y_test = y_test[pos] 
    studyid_test = studyid_test[pos]

    return X_test, y_test, studyid_test

