from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier, BayesianRidge
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, DotProduct, WhiteKernel
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np

RANDOM_SEED = 1234


model_configs = {
    

    "XGBoostBaseline": {
        "model": XGBClassifier,
        "model_hyperparams": {            
        },
        "param_grid": {

        },
    },

    "XGBoostGrid": {
        "model": XGBClassifier,
        "model_hyperparams": {               
        },
        "param_grid": {
            "model__gamma":[ 0.0, 0.2, 0.5, 1, 2, 5,10, 20],
            "model__max_depth": [3, 6, 9 ,12],
            "model__min_child_weight": [ 0, 1, 3, 5, 7],
            "model__learning_rate": [0.05, 0.10, 0.2],     
            "model__objective":[ "multi:softmax", "multi:softprob"],   
            "model__eval_metric":["auc","aucpr", "pre"]      
        },
    },


    "XGBoostOpt": {
        "model": XGBClassifier,
        "model_hyperparams": {   
            'eval_metric': 'auc', 
            'gamma': 0.0, 
            'learning_rate': 0.2, 
            'max_depth': 12, 
            'min_child_weight': 0, 
            'objective': 'multi:softmax'                        
        },
        "param_grid": {
            
        },
    },


    "GradientBoost": {
        "model": GradientBoostingClassifier,
        "model_hyperparams": {},
        "param_grid": {
            "model__n_estimators": [100, 500,],
            "model__learning_rate": [0.01,0.1,0.2],
            "model__subsample": [0.3,0.8,1.0],
            "model__max_depth": [3,8],
            "model__max_features": [0.5,1.0],
            "model__criterion": ["friedman_mse",  "squared_error"],
            "model__subsample":[0.4,  0.7, 1.0]
        },
    },

    "CatBoost": {
        "model": CatBoostClassifier,
        "model_hyperparams": {

        },
        "param_grid": {
            "model__max_depth": [6, 8, 10],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__l2_leaf_reg": [1, 5, 10],
        },
    },

    "AdaBoost": {
        "model": AdaBoostClassifier,
        "model_hyperparams": {

        },
        "param_grid": {
        "model__n_estimators": [50, 100,300,600],
        "model__learning_rate":[0.8,1,2,4,8],
        "model__algorithm": ["SAMME","SAMME.R"],
        "model__estimator": [DecisionTreeClassifier(), SVC(kernel=RationalQuadratic()), SGDClassifier(), BayesianRidge()],
        },
    },

    "LGBMClassifier": {
        "model": LGBMClassifier,
        "model_hyperparams": {
        },
        "param_grid": {
            "model__n_estimators":[50, 100 , 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "model__max_depth": [5, 6, 7, 8, 9],
            "model__num_leaves": [32, 64, 128, 256], 
            "model__min_data_in_leaf": [10, 100, 500, 1000], 
        },
    },

    "HistGradientBoostingClassifier": {
        "model": HistGradientBoostingClassifier,
        "model_hyperparams": {
        },
        "param_grid": {
            "model__l2_regularization":[0.001, 0.01 , 0.1],
            "model__learning_rate": [0.01, 0.1, 1, 10],
            "model__max_leaf_nodes": [3, 10, 30],
            "model__max_depth": [3, 7, 12, 18], 
        },
    },

    "RandomForestClassifier": {
        "model": RandomForestClassifier,
        "model_hyperparams": {
        },
        "param_grid": {
            "model__bootstrap":[True, False],
            "model__max_features": ['auto', 'sqrt'],
            "model__min_samples_leaf": [1, 2, 4],
            "model__min_samples_split": [2,5,10],
            "model__n_estimators":[200,400,600],
            "model__max_depth": [10, 20, 30, 40, 50], 
        },
    },


    "GaussianProcessClassifier": {
        "model": GaussianProcessClassifier,
        "model_hyperparams": {
        },
        "param_grid": {
            "model__n_restarts_optimizer": [1, 2, 3, 4, 5, 6, 8, 10, 14],
            "model__kernel": [RationalQuadratic(), RBF(), DotProduct(), WhiteKernel()],
        },
    },

    "StackingClassifier": {
        "model": StackingClassifier,
        "model_hyperparams": {
            "final_estimator": CatBoostClassifier(l2_leaf_reg = 5, 
                                                          learning_rate = 0.2, 
                                                          max_depth = 10),
            "cv": 5,
            "estimators": [
                ("XGBClassifier", XGBClassifier(eval_metric = "auc", 
                                                gamma = 0.0, 
                                                learning_rate = 0.2, 
                                                max_depth = 12, 
                                                min_child_weight = 0, 
                                                objective = "multi:softmax"  )),

                ("GradientBoostingClassifier", GradientBoostingClassifier(criterion = 'friedman_mse',
                                                                        learning_rate = 0.1,
                                                                        max_depth = 3, 
                                                                        max_features = 1.0, 
                                                                        n_estimators = 500,
                                                                        subsample = 1.0)),

                ("CatBoostClassifier", CatBoostClassifier(l2_leaf_reg = 5, 
                                                          learning_rate = 0.2, 
                                                          max_depth = 10)),

                ("AdaBoostClassifier", AdaBoostClassifier(algorithm ="SAMME",
                                                          estimator = SGDClassifier(), 
                                                          learning_rate = 2,
                                                          n_estimators = 50)),

                                                          
                ("LGBMClassifier", LGBMClassifier(learning_rate = 0.3,
                                                  max_depth = 6, 
                                                  min_data_in_leaf  = 100, 
                                                  n_estimators = 500, 
                                                  num_leaves = 32)),
                
                
                ("HistGradientBoostingClassifier", HistGradientBoostingClassifier(l2_regularization = 0.001,
                                                                                  learning_rate = 0.1, 
                                                                                  max_depth = 12, 
                                                                                  max_leaf_nodes = 30)),

                ("RandomForestClassifier", RandomForestClassifier(bootstrap = False, 
                                                                  max_depth = 30, 
                                                                  max_features = 'sqrt', 
                                                                  min_samples_leaf = 1, 
                                                                  min_samples_split = 2, 
                                                                  n_estimators= 400)                    
                ),

                ("GaussianProcessClassifier", GaussianProcessClassifier(kernel = DotProduct(sigma_0=1), 
                                                                        n_restarts_optimizer = 14))

            ]
            },
        
        "param_grid": {},

    },






}


            