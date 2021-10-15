import pandas as pd
from random import sample
import os
import pickle
import numpy as np
from util.run_trail import trail
from util.get_result import auc_summary, get_feature_names, feature_imp_summary


def test_trail():
    method = "xgb"
    for visit in range(1, 5):

        output_directory = "result/visit{}/{}/".format(visit, method)
        params = {"path": "data/V{}.csv".format(visit),
                "clf": method,
                "gridsearch": False,
                "feature_importance": True,
                "n_trails": 20,
                "n_splits": 5,
                "test_size": 0.2,
                "ignore_features": ["PublicID", "OUTCOME"],
                "output":output_directory
                }        

        
        trail(**params)
        
        auc_summary(output_directory, "result/summary/AUC/",  "{}_{}_AUC.csv".format(method, visit))

        features = get_feature_names(params["path"], pca_feat = None, n_comp = None)

        # get  feature importance
        if method in ["xgb", "RF"]:
            output_directory = "result/visit{}/{}/".format(visit, method)
            feature_imp_summary(params["output"], "result/summary/FI/", features, "V{}.csv".format(visit))

if __name__ == "__main__":

    test_trail()
 