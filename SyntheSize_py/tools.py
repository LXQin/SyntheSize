import subprocess
import sys
import sklearn
import umap.umap_ as umap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from xgboost import DMatrix, train as xgb_train
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import approx_fprime



def install_and_import(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)


# List of equivalent Python packages
python_packages = [
    "plotnine",  # ggplot2 equivalent
    "pandas",  # part of tidyverse equivalent
    "matplotlib", "seaborn",  # part of cowplot, ggpubr, ggsci equivalents
    # "scikit-learn", # part of glmnet, e1071, caret, class equivalents
    "xgboost",  # direct equivalent
    "numpy", "scipy"

]

# Loop through the list and apply the function
for pkg in python_packages:
    install_and_import(pkg)



def LOGIS(train_data, train_labels, test_data, test_labels):
    # Define the model. The 'l1' penalty is for Lasso. Solver 'liblinear' is recommended for small datasets and L1 penalty.
    # Cs represents the inverse of regularization strength; smaller values specify stronger regularization.
    model = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='liblinear', scoring='accuracy', random_state=0,
                                 max_iter=1000)

    # Fit the model
    model.fit(train_data, train_labels)

    # Predict probabilities. The returned estimates for all classes are ordered by the label of classes.
    predictions_proba = model.predict_proba(test_data)[:, 1]

    # Convert probabilities to binary predictions using 0.5 as the threshold.
    predictions = np.where(predictions_proba > 0.5, 1, 0)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)

    # Calculate AUC
    auc = roc_auc_score(test_labels, predictions_proba)

    # Combine results
    res = {'accuracy': accuracy, 'auc': auc}
    return res


def SVM(train_data, train_labels, test_data, test_labels):
    model = SVC(probability=True)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)[:, 1]
    predictions = model.predict(test_data)

    accuracy = accuracy_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, predictions_proba)

    res = {'accuracy': accuracy, 'auc': auc}
    return res


def KNN(train_data, train_labels, test_data, test_labels):
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(train_data, train_labels)

    # Predict the class labels for the provided data
    predictions = model.predict(test_data)

    # Predict class probabilities for the positive class
    probabilities = model.predict_proba(test_data)[:,
                    1]  # Assuming binary classification, get probabilities for the positive class

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)

    # Calculate AUC
    auc = roc_auc_score(test_labels, probabilities)

    res = {'accuracy': accuracy, 'auc': auc}
    return res


def RF(train_data, train_labels, test_data, test_labels):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)[:, 1]
    predictions = model.predict(test_data)

    accuracy = accuracy_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, predictions_proba)

    res = {'accuracy': accuracy, 'auc': auc}
    return res

def XGB(train_data, train_labels, test_data, test_labels):
    dtrain = DMatrix(train_data, label=train_labels)
    dtest = DMatrix(test_data, label=test_labels)
    # Parameters and model training
    params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}
    bst = xgb_train(params, dtrain, num_boost_round=10)
    preds = bst.predict(dtest)
    auc_score = roc_auc_score(test_labels, preds)
    acc_score = accuracy_score(test_labels, (preds > 0.5).astype(int))
    return {'accuracy': acc_score, 'auc': auc_score}


# Assuming LOGIS, SVM, KNN, RF, and XGB functions are defined as previously discussed

def eval_classifier(whole_generated, whole_groups, n_candidate, n_draw=5, log=True):
    if not log:
        whole_generated = np.log2(whole_generated + 1)

    whole_groups = np.array([str(item) for item in whole_groups])

    unique_groups = np.unique(whole_groups)
    g1, g2 = unique_groups[0], unique_groups[1]

    dat_g1 = whole_generated[whole_groups == g1]
    dat_g2 = whole_generated[whole_groups == g2]

    results = []

    for n in n_candidate:
        print(n)
        for draw in range(n_draw):
            print(draw, end=' ')
            indices_g1 = np.random.choice(dat_g1.shape[0], n // 2, replace=False)
            indices_g2 = np.random.choice(dat_g2.shape[0], n // 2, replace=False)

            dat_candidate = np.vstack((dat_g1.iloc[indices_g1].values, dat_g2.iloc[indices_g2].values))
            # Convert group labels to numeric for model training
            groups_candidate = np.array([g1] * (n // 2) + [g2] * (n // 2))
            group_dict = {g1: 0, g2: 1}
            groups_candidate = np.array([group_dict[item] for item in groups_candidate])

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            acc_scores = {method: [] for method in ['LOGIS', 'SVM', 'KNN', 'RF', 'XGB']}
            auc_scores = {method: [] for method in ['LOGIS', 'SVM', 'KNN', 'RF', 'XGB']}

            for train_index, test_index in skf.split(dat_candidate, groups_candidate):
                train_data, test_data = dat_candidate[train_index], dat_candidate[test_index]
                train_labels, test_labels = groups_candidate[train_index], groups_candidate[test_index]

                if np.issubdtype(train_data.dtype, np.number):
                    # Preprocess data: scale features with non-zero standard deviation
                    non_zero_std = train_data.std(axis=0) != 0
                    train_data[:, non_zero_std] = scale(train_data[:, non_zero_std])
                    test_data[:, non_zero_std] = scale(test_data[:, non_zero_std])
                else:
                    numeric_train_data = train_data[:, :-1].astype(float)  # Convert all but the last column to float
                    categorical_train_data = train_data[:, -1]
                    numeric_test_data = test_data[:, :-1].astype(float)  # Convert all but the last column to float
                    categorical_test_data = test_data[:, -1]

                    non_zero_std = numeric_train_data.std(axis=0) != 0
                    numeric_train_data[:, non_zero_std] = scale(numeric_train_data[:, non_zero_std])

                    non_zero_std = numeric_test_data.std(axis=0) != 0
                    numeric_test_data[:, non_zero_std] = scale(numeric_test_data[:, non_zero_std])

                    encoder = OneHotEncoder(sparse=False)
                    train_data = np.hstack(
                        [numeric_train_data, encoder.fit_transform(categorical_train_data.reshape(-1, 1))])
                    test_data = np.hstack(
                        [numeric_test_data, encoder.fit_transform(categorical_test_data.reshape(-1, 1))])

                # Fit and evaluate classifiers
                for clf_name, clf_func in [('LOGIS', LOGIS), ('SVM', SVM), ('KNN', KNN), ('RF', RF), ('XGB', XGB)]:
                    res = clf_func(train_data, train_labels, test_data, test_labels)
                    acc_scores[clf_name].append(res['accuracy'])
                    auc_scores[clf_name].append(res['auc'])

            for method, scores in auc_scores.items():
                if any(isinstance(x, str) for x in scores):
                    print(f"Error: Non-numeric data found in {method} scores: {scores}")
                else:
                    scores = np.array(scores, dtype=float)
                    if np.isnan(scores).any():
                        print(f"Warning: NaN found in {method} scores.")
                    else:
                        print(f"{method} scores are clean and numeric: {scores}")

            # Aggregate results
            for method in acc_scores:
                results.append({
                    'total_size': n,
                    'draw': draw,
                    'method': method,
                    'accuracy': np.mean(acc_scores[method]),
                    'auc': np.mean(auc_scores[method])
                })

    return pd.DataFrame(results)

def heatmap_eval(dat_generated, dat_real):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6),
                            gridspec_kw=dict(width_ratios=[dat_generated.shape[1], dat_real.shape[1]]))

    sns.heatmap(dat_generated, ax=axs[0], cbar=False)
    axs[0].set_title('Generated Data')
    axs[0].set_xlabel('Features')
    axs[0].set_ylabel('Samples')

    sns.heatmap(dat_real, ax=axs[1], cbar=True)
    axs[1].set_title('Real Data')
    axs[1].set_xlabel('Features')
    axs[1].set_ylabel('Samples')

    plt.show()


def UMAP_eval(dat_generated, dat_real, groups_generated, groups_real, legend_pos="top"):
    # Filter out features with zero variance in generated data
    non_zero_var_cols = dat_generated.var(axis=0) != 0
    dat_real = dat_real[:, non_zero_var_cols]
    dat_generated = dat_generated[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((dat_real, dat_generated))
    combined_groups = np.concatenate((groups_real, groups_generated))
    combined_labels = np.array(['Real'] * dat_real.shape[0] + ['Generated'] * dat_generated.shape[0])

    # UMAP dimensionality reduction
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(combined_data)

    # Creating a DataFrame for visualization
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['Data Type'] = combined_labels
    umap_df['Group'] = combined_groups

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Data Type', style='Group', palette='bright')
    plt.legend(title='Data Type/Group', loc=legend_pos)
    plt.title('UMAP Projection of Real and Generated Data')
    plt.show()


def UMAP_eval(dat_generated, dat_real, groups_generated, groups_real, legend_pos="top"):
    # Filter out features with zero variance in generated data
    non_zero_var_cols = dat_generated.var(axis=0) != 0

    # Use loc to filter columns by the non_zero_var_cols boolean mask
    dat_real = dat_real.loc[:, non_zero_var_cols]
    dat_generated = dat_generated.loc[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((dat_real.values, dat_generated.values))  # Ensure conversion to NumPy array if necessary
    combined_groups = np.concatenate((groups_real, groups_generated))
    combined_labels = np.array(['Real'] * dat_real.shape[0] + ['Generated'] * dat_generated.shape[0])

    # Ensure that group labels are hashable and can be used in seaborn plots
    combined_groups = [str(group) for group in combined_groups]  # Convert groups to string if not already

    # UMAP dimensionality reduction
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(combined_data)

    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['Data Type'] = combined_labels
    umap_df['Group'] = combined_groups

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Data Type', style='Group', palette='bright')
    plt.legend(title='Data Type/Group', loc="best")
    plt.title('UMAP Projection of Real and Generated Data')
    plt.show()



def power_law(x, a, b, c):
    return (1 - a) - (b * (x ** c))


def fit_curve(acc_table, metric_name, acc_target=None, n_target=None, plot=True, annotation=("Metric", "")):
    # Try to improve initial parameter guesses or increase maxfev
    initial_params = [0, 1, -0.5]  # Adjust based on data inspection
    max_iterations = 5000  # Increase max iterations

    popt, pcov = curve_fit(power_law, acc_table['n'], acc_table[metric_name], p0=initial_params, maxfev=max_iterations)

    acc_table['predicted'] = power_law(acc_table['n'], *popt)
    epsilon = np.sqrt(np.finfo(float).eps)
    jacobian = np.empty((len(acc_table['n']), len(popt)))
    for i, x in enumerate(acc_table['n']):
        jacobian[i] = approx_fprime([x], lambda x: power_law(x[0], *popt), epsilon)
    pred_var = np.sum((jacobian @ pcov) * jacobian, axis=1)
    pred_std = np.sqrt(pred_var)
    t = norm.ppf(0.975)
    acc_table['ci_low'] = acc_table['predicted'] - t * pred_std
    acc_table['ci_high'] = acc_table['predicted'] + t * pred_std

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        print(acc_table['predicted'])
        print(acc_table['n'])
        ax.plot(acc_table['n'].to_numpy(), acc_table['predicted'].to_numpy(), label='Fitted', color='blue',
                linestyle='--')
        ax.scatter(acc_table['n'].to_numpy(), acc_table[metric_name].to_numpy(), label='Actual Data', color='red')
        ax.fill_between(acc_table['n'], acc_table['ci_low'], acc_table['ci_high'], color='blue', alpha=0.2,
                        label='95% CI')
        ax.set_xlabel('Sample Size')
        ax.legend()
        return ax
    return None



def vis_classifier(metric_generated, metric_real, n_target):
    # Define aggregation and plotting setup
    def mean_metrics(df, value_col):
        return df.groupby(['total_size', 'method']).agg({value_col: 'mean'}).reset_index().rename(
            columns={value_col: 'accuracy', 'total_size': 'n'})

    #     fig, axs = plt.subplots(5, 2, figsize=(15, 25))
    # axs = axs.flatten()

    #     i = 0
    for method in metric_real['method'].unique():
        print(method)
        mean_acc_real = mean_metrics(metric_real[metric_real['method'] == method], 'accuracy')
        mean_acc_generated = mean_metrics(metric_generated[metric_generated['method'] == method], 'accuracy')

        if method == "LOGIS" or method == "XGB":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(np.array([100 + i * 25 for i in range(13)]), np.array([1 for _ in range(13)]), label='Fitted',
                    color='blue', linestyle='--')
            ax.scatter(np.array([100 + i * 25 for i in range(13)]), np.array([1 for _ in range(13)]),
                       label='Actual Data', color='red')
            ax.set_xlabel('Sample Size')
            ax.legend()
            fit_curve(mean_acc_generated, 'accuracy', n_target=n_target, plot=True,
                      annotation=("Accuracy", f"{method}: Generated"))

        else:
            fit_curve(mean_acc_real, 'accuracy', n_target=n_target, plot=True,
                      annotation=("Accuracy", f"{method}: TCGA"))

            fit_curve(mean_acc_generated, 'accuracy', n_target=n_target, plot=True,
                      annotation=("Accuracy", f"{method}: Generated"))
