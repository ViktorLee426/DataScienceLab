from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


from .feature_extractors import (
    RocketFeatureExtractorNoOverlapping,
    RocketFeatureExtractorOverlapping,
    VAEFeatureExtractorOverlapping, 
    VAEFeatureExtractor2,
    LSTMFeatureExtractor, 
    CNNFeatureExtractor,
)

preprocessing_configs = {


    "RocketFeatureExtractorOverlapping": {
        # Default baseline preprocessing
        "order": ["feature_extractor",  "scaler"],
        "feature_extractor": RocketFeatureExtractorOverlapping,
        "feature_extractor_hyperparams": {},
        "scaler": StandardScaler,
        "scaler_hyperparams": {},

        "param_grid": {},
    },

    "RocketFeatureExtractorNoOverlapping": {
        # Default baseline preprocessing
        "order": ["feature_extractor",  "scaler"],
        "feature_extractor": RocketFeatureExtractorNoOverlapping,
        "feature_extractor_hyperparams": {},
        "scaler": StandardScaler,
        "scaler_hyperparams": {},

        "param_grid": {},
    },


    "VAEFeatureExtractorOverlapping": {
        # Default baseline preprocessing
        "order": ["feature_extractor", "imputer", "scaler", "selector"],
        "feature_extractor": VAEFeatureExtractorOverlapping,
        "feature_extractor_hyperparams": {},
        "imputer": SimpleImputer,
        "imputer_hyperparams": {"strategy": "median"},
        "scaler": StandardScaler,
        "scaler_hyperparams": {},
        "selector": SelectKBest,
        "selector_hyperparams": {
            "score_func": f_classif,
            "k": 100,
        },
        "param_grid": {},
    },
    
    "VAEFeatureExtractor2": {
        # Default baseline preprocessing
        "order": ["feature_extractor", "scaler"],
        "feature_extractor": VAEFeatureExtractor2,
        "feature_extractor_hyperparams": {},

        "scaler": StandardScaler,
        "scaler_hyperparams": {},

        "param_grid": {},
    },

    "LSTMFeatureExtractor": {
        # Default baseline preprocessing
        "order": ["feature_extractor", "scaler"],
        "feature_extractor": LSTMFeatureExtractor,
        "feature_extractor_hyperparams": {},

        "scaler": RobustScaler,
        "scaler_hyperparams": {},

        "param_grid": {},
    },

    "CNNFeatureExtractor": {
        # Default baseline preprocessing
        "order": ["feature_extractor", "scaler"],
        "feature_extractor": CNNFeatureExtractor,
        "feature_extractor_hyperparams": {},

        "scaler": RobustScaler,
        "scaler_hyperparams": {},

        "param_grid": {},
    },






}

