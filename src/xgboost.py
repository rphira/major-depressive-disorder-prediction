import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from optuna.integration import XGBoostPruningCallback
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from xgboost.callback import EarlyStopping

def objective(trial, X_train, y_train, X_test, y_test):
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'verbosity': 1,
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0),
    }

    model = xgb.XGBClassifier(
        **param,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    # Train the model without early stopping
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy

def loso_optuna_xg(subjects_features, subjects_labels, n_trials, save_directory):
    """
    Performs Leave-One-Subject-Out (LOSO) Cross-Validation with Optuna hyperparameter tuning using XGBoost,
    and monitors training and validation accuracy and loss over time for the best model only.

    Args:
        subjects_features (dict): Dictionary containing feature matrices for each subject.
        subjects_labels (dict): Dictionary containing label arrays for each subject.
        n_trials (int): Number of trials to perform for Optuna.
        save_directory (str): Directory where trained models will be saved.

    Returns:
        accuracies (dict): A dictionary with subjects as keys and accuracy as values.
    """
    # Ensure save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    accuracies = {}
    models = {}
    all_y_true = []
    all_y_pred = []

    label_encoder = LabelEncoder()
    all_labels = np.hstack([subjects_labels[subject] for subject in subjects_labels.keys()])
    label_encoder.fit(all_labels)

    for test_subject in subjects_features.keys():
        print(f"Leaving out {test_subject} as the test subject.")

        # Prepare training data
        X_train, y_train = [], []
        for subject, features in subjects_features.items():
            if subject != test_subject:
                X_train.append(features)
                y_train.append(subjects_labels[subject])

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        y_train = label_encoder.transform(y_train)

        # Test data from the left-out subject
        X_test = subjects_features[test_subject]
        y_test = subjects_labels[test_subject]
        y_test = label_encoder.transform(y_test)

        # Run Optuna optimization for hyperparameter tuning
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)

        # Get the best hyperparameters from the last trial (best trial)
        best_params = study.best_params
        best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric=['mlogloss', 'merror'], early_stopping_rounds=5)

        # Set up evaluation sets to monitor training and validation
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        # Train the best model and store evaluation results for the final trial
        best_model.fit(X_train, y_train, eval_set=eval_set)

        results = best_model.evals_result()

        # Save the trained model to a specific directory
        model_filename = os.path.join(save_directory, f"{test_subject}_model.joblib")  # Full path
        joblib.dump(best_model, model_filename)

        # Evaluate the best model on the test subject
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[test_subject] = accuracy

        print(f"Best hyperparameters for {test_subject}: {best_params}")
        print(f"Accuracy for {test_subject}: {accuracy:.2f}")

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Plot the training and validation accuracy and loss for the final (best) model
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(epochs)

        # Plot Log Loss
        plt.figure(figsize=(12, 6))
        plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
        plt.title(f'XGBoost Log Loss - {test_subject}')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.savefig(f'{save_directory}/{test_subject}_logloss.png')
        plt.close()

        # Plot classification error (merror)
        plt.figure(figsize=(12, 6))
        plt.plot(x_axis, results['validation_0']['merror'], label='Train')
        plt.plot(x_axis, results['validation_1']['merror'], label='Validation')
        plt.title(f'XGBoost Classification Error - {test_subject}')
        plt.xlabel('Epochs')
        plt.ylabel('Classification Error')
        plt.legend()
        plt.savefig(f'{save_directory}/{test_subject}_merror.png')
        plt.close()

    # Overall accuracy and classification report
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"\nOverall Accuracy across all subjects: {overall_accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=['Class 1', 'Class 2', 'Class 3']))

    return accuracies, models

# def loso_optuna_xg_OLD(subjects_features, subjects_labels, n_trials=5, save_directory='models/'):
#     """
#     Performs Leave-One-Subject-Out (LOSO) Cross-Validation with Optuna hyperparameter tuning using XGBoost.

#     Args:
#         subjects_features (dict): Dictionary containing feature matrices for each subject.
#         subjects_labels (dict): Dictionary containing label arrays for each subject.
#         n_trials (int): Number of trials to perform for Optuna.
#         save_directory (str): Directory where trained models will be saved.

#     Returns:
#         accuracies (dict): A dictionary with subjects as keys and accuracy as values.
#     """
#     # Ensure save directory exists
#     import os
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)

#     accuracies = {}
#     models = {}
#     all_y_true = []
#     all_y_pred = []

#     label_encoder = LabelEncoder()
#     all_labels = np.hstack([subjects_labels[subject] for subject in subjects_labels.keys()])
#     label_encoder.fit(all_labels)

#     for test_subject in subjects_features.keys():
#         print(f"Leaving out {test_subject} as the test subject.")

#         # Prepare training data
#         X_train, y_train = [], []
#         for subject, features in subjects_features.items():
#             if subject != test_subject:
#                 X_train.append(features)
#                 y_train.append(subjects_labels[subject])

#         X_train = np.vstack(X_train)
#         y_train = np.hstack(y_train)
#         y_train = label_encoder.transform(y_train)

#         # Test data from the left-out subject
#         X_test = subjects_features[test_subject]
#         y_test = subjects_labels[test_subject]
#         y_test = label_encoder.transform(y_test)

#         # Run Optuna optimization for hyperparameter tuning
#         study = optuna.create_study(direction='maximize')
#         study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)

#         # Get the best hyperparameters
#         best_params = study.best_params
#         best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric=['mlogloss', 'merror'])
#         best_model.fit(X_train, y_train)

#         # Save the trained model to a specific directory
#         model_filename = os.path.join(save_directory, f"{test_subject}_model.joblib")  # Full path
#         joblib.dump(best_model, model_filename)

#         # Evaluate the best model on the test subject
#         y_pred = best_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         accuracies[test_subject] = accuracy

#         print(f"Best hyperparameters for {test_subject}: {best_params}")
#         print(f"Accuracy for {test_subject}: {accuracy:.2f}")

#         all_y_true.extend(y_test)
#         all_y_pred.extend(y_pred)

#     overall_accuracy = accuracy_score(all_y_true, all_y_pred)
#     print(f"\nOverall Accuracy across all subjects: {overall_accuracy:.2f}")

#     print("\nClassification Report:")
#     print(classification_report(all_y_true, all_y_pred, target_names=['Class 1', 'Class 2', 'Class 3']))

#     return accuracies, models  # Return accuracies and models