from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from feature_extractor import featuresExtractor
from general.helpers import divide_dataset, create_method_name, create_kfold_matrix, create_bin_task_labels
import tqdm
from multiprocessing import Pool
from functools import partial
import pickle

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


def build_multiclass_experiment(dataset, feats, learner_name, focus, output_path, pickle_path):
    """
    Build the multiclass experiment for classic ML methods and save the results.
    :param dataset: the dataset
    :param feats: the features to extract
    :param learner_name: name of the learner to use (svm, lr, rf)
    :param focus: the focus of the classification (name, wing, party, gender, birth, party_status)
    :param output_path: file path for the csv file with best parameters, tot_features, macro and micro values
    :param pickle_path: file path for the pickle file with the labels (true and predictions)
    """
    if os.path.exists(output_path) and os.path.exists(pickle_path):
        df_csv = pd.read_csv(output_path, sep=';')
        with open(pickle_path, 'rb') as pickle_file:
            df_preds = pickle.load(pickle_file)
    else:
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
        df_csv = pd.DataFrame(columns=['Method', 'Best_params', 'Tot_features', 'Macro_F1', 'micro_F1'])
        df_preds = {}
    method_name = create_method_name(feats)
    if method_name in df_csv['Method'].values and method_name in df_preds:
        print(f'MulticlassClassification {learner_name} experiment {method_name} already done!')
    else:
        print(f'----- MulticlassClassification {learner_name} experiment {method_name} -----')
        trval_data, te_data = divide_dataset(dataset, ratio='50_test', focus=focus)
        y_pred, y_te, best_params, tot_features = _optimization_classification(learner_name, trval_data, te_data, feats, score='macro')
        if 'True' not in df_preds:
            df_preds['True'] = y_te
        df_preds[method_name] = y_pred
        macro_f1 = f1_score(y_te, y_pred, average='macro')
        micro_f1 = f1_score(y_te, y_pred, average='micro')
        row = {'Method': method_name, 'Best_params': best_params, 'Tot_features': int(np.round(tot_features)),
               'Macro_F1': np.around(macro_f1, decimals=3), 'micro_F1': np.around(micro_f1, decimals=3)}
        df_csv = df_csv.append(row, ignore_index=True)
    print('----- THE END -----')
    print('Tot. features:', df_csv[df_csv['Method'] == method_name]['Tot_features'].values[0])
    print(f'Macro-F1:', df_csv[df_csv['Method'] == method_name]['Macro_F1'].values[0])
    print(f'micro-F1:', df_csv[df_csv['Method'] == method_name]['micro_F1'].values[0])
    df_csv.to_csv(path_or_buf=output_path, sep=';', index=False, header=True)
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(df_preds, pickle_file)


def build_binary_experiment(dataset, feats, learner_name, focus, output_path, pickle_path):
    """
    Build the binary experiment for classic ML methods and save the results.
    :param dataset: the dataset
    :param feats: the features to extract
    :param learner_name: name of the learner to use (svm, lr, rf)
    :param focus: the focus of the classification (name, wing, party, gender, birth, party_status)
    :param output_path: file path for the csv file with F1 values
    :param pickle_path: file path for the pickle file with the labels (true and predictions)
    """
    if focus == 'party_status':
        unique_labels = dataset['Party_status'].unique()
    else:
        unique_labels = dataset[f'Speaker_{focus}'].unique()
    if os.path.exists(output_path):
        df_csv = pd.read_csv(output_path, sep=';')
        with open(pickle_path, 'rb') as pickle_file:
            df_preds = pickle.load(pickle_file)
    else:
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
        columns = ['Method']
        columns.extend(unique_labels)
        df_csv = pd.DataFrame(columns=columns)
        df_preds = {}
    method_name = create_method_name(feats)
    if method_name in df_csv['Method'].values:
        print(f'BinaryClassification {learner_name} experiment {method_name} already done!')
    else:
        print(f'----- BinaryClassification {learner_name} experiment {method_name} -----')
        trval_data, te_data = divide_dataset(dataset, ratio='50_test', focus=focus)
        f1s = []
        labels_best_models = []
        for label in unique_labels:
            print(f'Experiment for {label}')
            if label not in df_preds:
                df_preds[label] = {}
            label_trval_data = create_bin_task_labels(trval_data, label)
            label_te_data = create_bin_task_labels(te_data, label)
            y_pred, y_te, best_model, _ = _optimization_classification(learner_name, label_trval_data, label_te_data, feats, score='binary')
            labels_best_models.append((label, best_model))
            if 'True' not in df_preds[label]:
                df_preds[label]['True'] = y_te
            df_preds[label][method_name] = y_pred
            f1 = np.around(f1_score(y_te, y_pred, average='binary'), decimals=3)
            print(f'F1: {f1}\n')
            f1s.append((label, f1))
        row = {'Method': method_name}
        for label, f1 in f1s:
            row[label] = f1
        df_csv = df_csv.append(row, ignore_index=True)
    print('----- THE END -----')
    print(f'F1 per author:')
    for label in unique_labels:
        print(label, df_csv[df_csv['Method'] == method_name][label].values[0])
    df_csv.to_csv(path_or_buf=output_path, sep=';', index=False, header=True)
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(df_preds, pickle_file)


# optimization of hyper-parameters via GridSearch and k-fold cross-validation (from scratch) on the train set
# and classification of the test set
def _optimization_classification(learner_name, trval_data, te_data, feats, score):
    n_splits = 5
    if learner_name == 'lr':
        params = {'class_weight': ['balanced', None], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [1000],
                  'random_state': [42]}
    elif learner_name == 'rf':
        params = {'class_weight': ['balanced', None], 'n_estimators': [10, 50, 100, 150, 200, 350, 500],
                  'criterion': ['gini', 'entropy'], 'random_state': [42]}
    else:
        params = {'class_weight': ['balanced', None], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'random_state': [42]}
    params_grid = ParameterGrid(params)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = kfold.split(trval_data['texts'], trval_data['y'])
    print('Creating train/val k-fold matrixes...')
    with Pool(processes=n_splits) as p:
        extraction_step = partial(create_kfold_matrix, trval_data, feats)
        X_trs, X_vals, y_trs, y_vals = zip(*p.imap(extraction_step, splits))
    print("Training shape (first matrix): ", X_trs[0].shape)
    print("Validation shape (first matrix): ", X_vals[0].shape)
    print('OPTIMIZATION')
    with Pool(processes=12) as p:
        optimization_step = partial(__kfold_experiment, X_trs, X_vals, y_trs, y_vals, learner_name, score)
        opt_results = list(tqdm.tqdm(p.imap(optimization_step, params_grid), total=len(params_grid)))
    best_result_idx = opt_results.index(max(opt_results, key=lambda result: result))
    print('Best model:', params_grid[best_result_idx])
    print('CLASSIFICATION')
    if learner_name == 'lr':
        cls = LogisticRegression(**params_grid[best_result_idx])
    elif learner_name == 'rf':
        cls = RandomForestClassifier(**params_grid[best_result_idx])
    else:
        cls = SVC(**params_grid[best_result_idx])
    print('Creating training+validation / test feature matrix...')
    X_trval, X_te, _ = featuresExtractor(trval_data, te_data, **feats)
    print("Training shape: ", X_trval.shape)
    print("Test shape: ", X_te.shape)
    cls.fit(X_trval, trval_data['y'])
    y_pred = cls.predict(X_te)
    return y_pred, te_data['y'], params_grid[best_result_idx], X_te.shape[1]


# perform kfold cross-validation for GridSearch optimization
def __kfold_experiment(X_trs, X_vals, y_trs, y_vals, learner_name, score, params_combination):
    if learner_name == 'lr':
        learner = LogisticRegression(**params_combination)
    elif learner_name == 'rf':
        learner = RandomForestClassifier(**params_combination)
    else:
        learner = SVC(**params_combination)
    preds = []
    for i in range(len(X_trs)):
        cls = learner.fit(X_trs[i], y_trs[i])
        preds.extend(cls.predict(X_vals[i]))
    result = f1_score(sum(y_vals, []), preds, average=score)
    return result


def party_status_experiment(dataset, feats, learner_name, output_path, pickle_path, author='Sánchez Pérez-castejón, Pedro'):
    statuses = dataset['Party_status'].unique()
    if os.path.exists(output_path):
        df_csv = pd.read_csv(output_path, sep=';')
        with open(pickle_path, 'rb') as pickle_file:
            df_preds = pickle.load(pickle_file)
    else:
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        columns = ['Method']
        columns.extend(statuses)
        df_csv = pd.DataFrame(columns=columns)
        df_preds = {}
    method_name = create_method_name(feats)
    if method_name in df_csv['Method'].values:
        print(f'BC {learner_name} experiment {method_name} already done!')
    else:
        f1s = []
        for status in statuses:
            print(f'---- BC {learner_name} experiment {method_name} ----')
            print(f'--- Party-status experiment for {author} when in {status} ---')
            if status not in df_preds:
                df_preds[status] = {}
            mini_dataset = dataset.drop(dataset[(dataset.Speaker_name == author) & (dataset.Party_status != status)].index)
            trval_data, te_data = divide_dataset(mini_dataset, ratio='50_test', focus='name')
            author_trval_data = create_bin_task_labels(trval_data, author)
            author_te_data = create_bin_task_labels(te_data, author)
            y_pred, y_te, _, _ = _optimization_classification(learner_name, author_trval_data, author_te_data, feats, score='binary')
            if 'True' not in df_preds[status]:
                df_preds[status]['True'] = y_te
            df_preds[status][method_name] = y_pred
            f1 = np.around(f1_score(y_te, y_pred, average='binary'), decimals=3)
            print(f'F1: {f1}\n')
            f1s.append((status, f1))
        row = {'Method': method_name}
        for status, f1 in f1s:
            row[status] = f1
        df_csv = df_csv.append(row, ignore_index=True)
    df_csv.to_csv(path_or_buf=output_path, sep=';', index=False, header=True)
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(df_preds, pickle_file)
