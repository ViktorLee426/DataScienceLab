import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import confusion_matrix, accuracy_score

from configs.model_configs import model_configs
from configs.preprocessing_configs import preprocessing_configs

# sklearn uses np.random. Keep in mind that np.random is not threadsafe
np.random.seed(3141592)

if __name__ == "__main__":
    # # Parse arguments, setup the experiment folder, load the configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pc",
        "--preprocessing_config",
        required=True,
        type=str,
        help="Name of the preprocessing config in configs/preprocessing_configs "
        "used for this experiment.",
    )
    parser.add_argument(
        "-mc",
        "--model_config",
        required=True,
        type=str,
        help="Name of the model config in configs/model_configs "
        "used for this experiment.",
    )
    parser.add_argument(
        "-cv", "--cv_folds", type=int, default=5, help="Number of folds to use in CV."
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs used for grid search. More than 1 uses multithreading.",
    )
    args = parser.parse_args()

    if args.preprocessing_config not in preprocessing_configs:
        raise ValueError(f"Unkown preprocessing config {args.preprocessing_config}.")
    if args.model_config not in model_configs:
        raise ValueError(f"Unkown model config {args.model_config}.")

    experiment_name = f"{args.preprocessing_config}--{args.model_config}"

    if os.path.isdir(f"results/{experiment_name}"):
        raise Exception(
            f"Results for experiment {experiment_name} already exist. Delete the "
            f"results/{experiment_name} folder to be able to redo the experiment."
        )

    p_config = preprocessing_configs[args.preprocessing_config]
    m_config = model_configs[args.model_config]

    # Preprocess the data into features
    feature_extractor = p_config["feature_extractor"](
        **p_config["feature_extractor_hyperparams"]
    )
    X_train, y_train, X_test, y_test = feature_extractor.load_data()

    # Create the Preprocessing + Model pipeline
    # Add preprocessing steps to the pipeline
    pipeline_steps = [
        (step, p_config[f"{step}"](**p_config[f"{step}_hyperparams"]))
        for step in p_config["order"]
    ]

    # Add model to the pipeline
    model = m_config["model"](**m_config["model_hyperparams"])
    pipeline_steps.append(("model", model))

    pipeline = Pipeline(steps=pipeline_steps)

    # # Grid Search CV
    param_grid = p_config["param_grid"]
    param_grid.update( m_config["param_grid"])  # Combine param_grids from p_config and m_config


    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        n_jobs=args.n_jobs,
        scoring= "accuracy",
        return_train_score=True,
        verbose=3,
        cv=args.cv_folds,
    )

    grid_search.fit(X_train, np.ravel(y_train))

    # # Process the results results
    print("################## Training Results ######################")
    print("Best Score: ", grid_search.best_score_)
    print("Best Estimator: ", grid_search.best_estimator_)
    print("Best Parameter Setting: ", grid_search.best_params_)

    cv_results = grid_search.cv_results_
    cv_results_df = pd.DataFrame(cv_results).T
    print("CrossValidation results:")
    print(cv_results_df)

    #Store the CV results in training set
    path_training_results = f"results/{experiment_name}/TrainingResults.txt"
    os.makedirs(os.path.dirname(path_training_results), exist_ok=True)
    with open(path_training_results, 'w') as file:
        file.write("Preprocessing Configuration:")
        file.write(str(p_config))
        file.write("\n Model Configuration:")
        file.write(str(m_config))
        file.write("\n Best Score: ")
        file.write(str(grid_search.best_score_))
        file.write("\n Best Estimator: ")
        file.write(str(grid_search.best_estimator_))
        file.write("\n Best Parameter Setting:") 
        file.write(str(grid_search.best_params_))
        file.write("\n")
        cv_results_df.to_csv(file, sep = "\t", index=True)
    
    #make predictions
    print("################## Testing Results ######################")
    
    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy:", acc)
    print("Confusion Matrix:")
    print(cm)

    #Store the testing results 
    path_testing_results = f"results/{experiment_name}/TestingResults.txt"
    os.makedirs(os.path.dirname(path_testing_results), exist_ok=True)
    with open(path_testing_results, 'w') as f:
        f.write("Accuracy:")
        f.write(str(acc))
        f.write("\n")
        f.write("Confusion Matrix:\n")
        for line in cm:
            f.write('\t'.join(str(x) for x in line) + '\n')






