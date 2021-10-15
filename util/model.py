from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from imblearn.under_sampling import TomekLinks

def get_classifier(gridsearch: bool , classifier: str):
    """ 
    Get the classifier to use classification. Support gridsearch but 
    only binary classification allows for gridsearch.
    
    Parameters 
        gridsearch (bool): True for grid search. Uses AUC score for grid search
        classifier (str): support [RF, SVM, DT, Bagging, LR] 

    Returns:
        pipe: sklearn classifier 
    """


    pipe, param_grid = get_model(classifier)

    if gridsearch: 
        scoring_metric = make_scorer(roc_auc_score)
        skf = StratifiedKFold(n_splits = 2)             
        grid = GridSearchCV(pipe, param_grid, 
                                    refit = True, 
                                    verbose = 1, 
                                    scoring=scoring_metric, 
                                    n_jobs=-1, 
                                    cv=skf) 
        return grid

    return pipe


def get_model(classifier: str):
    """
    Generate the model. If you want to mannually define the parameters in 
    classifier, e.g you can change the XXX in SVC(XXX). Also, please refer
    to pipeline package to see if you want add any preprocessing 
    """

    if classifier == "SVM":
        # this is use for gridsearch CV, given use the name of the model that we have define from 
        # the top {model_name}__{parameter}
        param_grid = {'SVC__C': [ 1, 10, 100],  
                      'SVC__gamma': [ 0.001, 0.0001], 
                      'SVC__kernel': ['linear','rbf']}  

        # you can change the model type here 
        # this use the pipeline from imbalance oversampling + StandardScaler + model 
        pipe = Pipeline([('under', RandomUnderSampler(sampling_strategy='majority')), ('SVC', SVC(C=1, gamma=0.0001, kernel='rbf',probability=True))])

    if classifier == "LR":
        param_grid = {'LR__C': [0.001, 0.1, 1, 10, 100, 1000],  
                     'LR__penalty': ['l2','l1']}  

        pipe = Pipeline([('under', RandomUnderSampler(sampling_strategy='majority')), ('LR', LogisticRegression())])

    if classifier == "DT":
        param_grid = {'DT__criterion': ['gini'],  
                      'DT__max_depth': [5]}      

        pipe = Pipeline([('under', RandomUnderSampler(sampling_strategy='majority')), ('DT', DecisionTreeClassifier(criterion='gini', max_depth=5))])

       
    if classifier == "RF":
        param_grid = {'RF__n_estimators': [150, 250, 300, 400],
                      'RF__max_features': [ 0.1, 0.2, 0.3, 0.4],
                      'RF__max_samples': [ 0.5, 0.6, 0.8],
                      'RF__min_samples_split': [0.1, 0.2, 0.3, 0.4]}

        pipe = Pipeline([('under', RandomUnderSampler(sampling_strategy='majority')),('RF', RandomForestClassifier(max_features=0.3, 
            max_samples = 0.8, n_estimators=400, min_samples_split=20, n_jobs=-1))])
    

    if classifier == "Bagging":
        param_grid = {}
        rfClf =  RandomForestClassifier(max_features=0.3, max_samples = 0.8, n_estimators=400, min_samples_split=20, n_jobs=-1)
        svmClf = SVC(C=1, gamma=0.0001, kernel='rbf',probability=True)
        logClf = LogisticRegression(C=0.001)
        pipe = Pipeline([ ('under', RandomUnderSampler(sampling_strategy='majority')), ("Esemble", VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf), ('log', logClf)], voting='soft')) ])
        
    
    if classifier == "xgb":
        param_grid = {}
        pipe = Pipeline([ ('under', RandomUnderSampler(sampling_strategy='majority')), 
                ("xgb", xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 10, alpha = 10, n_estimators = 400, 
                eval_metric='auc', use_label_encoder=False))])

    return pipe, param_grid












