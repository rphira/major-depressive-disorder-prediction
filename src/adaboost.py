import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib

def run_adaboost_LOSO(subjects_features, subjects_labels, y_true, y_pred):
    all_subjects = list(subjects_features.keys())
    
    for subject_out in all_subjects:
        # Prepare training data (all subjects except the one left out)
        train_features = []
        train_labels = []
        for subject in all_subjects:
            if subject != subject_out:
                train_features.append(subjects_features[subject])
                train_labels.append(subjects_labels[subject])
        
        # Combine the training data
        X_train = np.vstack(train_features)
        y_train = np.hstack(train_labels)
        
        # Prepare test data (the subject left out)
        X_test = subjects_features[subject_out]
        y_test = subjects_labels[subject_out]


        # Define AdaBoost classifier
        clf = AdaBoostClassifier()

        # Defining hyperparameters and range
        param_distributions = {
            'n_estimators': randint(50, 200),
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'base_estimator__max_depth': randint(1, 4)
        }
        
        # Perform random search with 5-fold cross-validation
        random_search = RandomizedSearchCV(clf, param_distributions, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
        random_search.fit(X_train, y_train)

        # Output the best parameters and the best score
        print("Best hyperparameters:", random_search.best_params_)
        print("Best score:", random_search.best_score_)

        
        # Predict on the left-out subject
        y_pred_single = clf.predict(X_test)
        
        # Collect predictions and true labels
        y_true.extend(y_test)
        y_pred.extend(y_pred_single)
        
        # Check progress
        print(f"Finished LOSO iteration for subject {subject_out}")
    
    # Evaluate overall accuracy across all LOSO iterations
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall accuracy: {accuracy:.4f}")
    overall_report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(overall_report)
