from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from feature_extractor import featuresExtractor
from general.sign_test import significance_test
from general.helpers import divide_dataset, create_method_name
import tqdm
from multiprocessing import Pool
from functools import partial
import pickle

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


# ------------------------------------------------------------------------
# method to do (classical ML) classification
# ------------------------------------------------------------------------
# saves the results in two different files:
# output_path = csv file with best parameters, tot_features, macro and micro values
# pickle_path = pickle file with labels (true or predictions)
def build_AA_experiment(dataset, feats, learner, output_path, pickle_path):
    assert learner in ['svm', 'lr', 'random_forest'], 'The learner must be svm, lr, or random_forest'
    if os.path.exists(output_path) and os.path.exists(pickle_path):
        df_csv = pd.read_csv(output_path, sep=';')
        with open(pickle_path, 'rb') as pickle_file:
            df_labels = pickle.load(pickle_file)
    else:
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
        if learner == 'lr':
            df_csv = pd.DataFrame(columns=['Method', 'Best_C', 'Tot_features', 'Macro_F1', 'micro_F1'])
        if learner == 'svm':
            df_csv = pd.DataFrame(columns=['Method', 'Best_C', 'Best_kernel', 'Tot_features', 'Macro_F1', 'micro_F1'])
        if learner == 'random_forest':
            df_csv = pd.DataFrame(columns=['Method', 'Best_n_estimators', 'Best_criterion', 'Tot_features', 'Macro_F1', 'micro_F1'])
        df_labels = {}
    method_name = create_method_name(feats)
    if method_name in df_csv['Method'].values and method_name in df_labels:
        print(f'{learner} experiment {method_name} already done!')
    else:
        print(f'----- {learner} experiment {method_name} -----')
        tr_data, val_data, te_data = divide_dataset(dataset)
        if learner == 'lr':
            y_pred, y_te, best_C, tot_features = _multi_classification(learner, tr_data, val_data, te_data, feats)
        if learner == 'svm':
            y_pred, y_te, best_C, best_kernel, tot_features = _multi_classification(learner, tr_data, val_data, te_data, feats)
        if learner == 'random_forest':
            y_pred, y_te, best_n_estimators, best_criterion, tot_features = _multi_classification(learner, tr_data, val_data, te_data, feats)
        if 'True' not in df_labels:
            df_labels['True'] = y_te
        df_labels[method_name] = y_pred
        macro_f1 = f1_score(y_te, y_pred, average='macro')
        micro_f1 = f1_score(y_te, y_pred, average='micro')
        if learner == 'lr':
            row = {'Method': method_name, 'Best_C': best_C, 'Tot_features': int(np.round(tot_features)),
                   'Macro_F1': np.around(macro_f1, decimals=3), 'micro_F1': np.around(micro_f1, decimals=3)}
        if learner == 'svm':
            row = {'Method': method_name, 'Best_C': best_C, 'Best_kernel': best_kernel,
                   'Tot_features': int(np.round(tot_features)),
                   'Macro_F1': np.around(macro_f1, decimals=3), 'micro_F1': np.around(micro_f1, decimals=3)}
        if learner == 'random_forest':
            row = {'Method': method_name, 'Best_n_estimators': best_n_estimators, 'Best_criterion': best_criterion,
                   'Tot_features': int(np.round(tot_features)),
                   'Macro_F1': np.around(macro_f1, decimals=3), 'micro_F1': np.around(micro_f1, decimals=3)}
        df_csv = df_csv.append(row, ignore_index=True)
    print('----- THE END -----')
    print('Tot. features:', df_csv[df_csv['Method'] == method_name]['Tot_features'].values[0])
    print(f'Macro-F1:', df_csv[df_csv['Method'] == method_name]['Macro_F1'].values[0])
    print(f'micro-F1:', df_csv[df_csv['Method'] == method_name]['micro_F1'].values[0])
    df_csv.to_csv(path_or_buf=output_path, sep=';', index=False, header=True)
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(df_labels, pickle_file)
    # significance tests for very other method
    if df_csv.shape[0] == 1:
        print(f'{method_name} is the only method stored, no significance test is performed')
    else:
        for index, row in df_csv.loc[df_csv['Method'] != method_name].iterrows():
            baseline = row['Method']
            print(f'COMPARISON WITH BASELINE {baseline}')
            delta_macro = (df_csv[df_csv['Method'] == method_name]['Macro_F1'].values[0] - row['Macro_F1']) / row['Macro_F1'] * 100
            delta_micro = (df_csv[df_csv['Method'] == method_name]['micro_F1'].values[0] - row['micro_F1']) / row['micro_F1'] * 100
            print(f'Macro-F1 Delta %: {delta_macro:.2f}')
            print(f'Micro-F1 Delta %: {delta_micro:.2f}')
            significance_test(df_labels['True'], df_labels[baseline], df_labels[method_name], baseline)


# optimization of parameters via GridSearch from scratch
def _multi_classification(learner, tr_data, val_data, te_data, feats):
    configurations = __create_GridSearch(learner)
    # create tr and val matrixes
    print('Creating training / validation feature matrix...')
    X_tr, X_val = featuresExtractor(tr_data, val_data, **feats)
    print("Training shape: ", X_tr.shape)
    print("Validation shape: ", X_val.shape)
    print('Parameter optimization...')
    with Pool(processes=6) as p:
        optimization_step = partial(__single_experiment, learner=learner, X_tr=X_tr, X_te=X_val, y_tr=tr_data['y'], y_te=val_data['y'])
        results = list(tqdm.tqdm(p.imap(optimization_step, configurations), total=len(configurations)))
    best_result_idx = results.index(max(results, key=lambda result: result))
    print('BEST MODEL')
    if learner in ['lr', 'svm']:
        if learner == 'lr':
            best_C = configurations[best_result_idx]
        else:
            best_C = configurations[best_result_idx][0]
            best_kernel = configurations[best_result_idx][1]
            print('Best Kernel:', best_kernel)
        print('Best C:', best_C)
    elif learner == 'random_forest':
        best_n_estimators = configurations[best_result_idx][0]
        best_criterion = configurations[best_result_idx][1]
        print('Best n_estimators:', best_n_estimators)
        print('Best criterion:', best_criterion)
    print(f'Best macro-f1: {results[best_result_idx]:.3f}')
    print('CLASSIFICATION')
    if learner == 'lr':
        cls = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, C=best_C)
    elif learner == 'svm':
        cls = SVC(class_weight='balanced', random_state=42, C=best_C, kernel=best_kernel)
    elif learner == 'random_forest':
        cls = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=best_n_estimators, criterion=best_criterion)
    trval_data = {key: np.concatenate((tr_data[key], val_data[key])) for key in tr_data}
    print('Creating training+validation / test feature matrix...')
    X_trval, X_te = featuresExtractor(trval_data, te_data, **feats)
    print("Training shape: ", X_trval.shape)
    print("Test shape: ", X_te.shape)
    cls.fit(X_trval, trval_data['y'])
    y_pred = cls.predict(X_te)
    if learner == 'lr':
        return y_pred, te_data['y'], best_C, X_te.shape[1]
    elif learner == 'svm':
        return y_pred, te_data['y'], best_C, best_kernel, X_te.shape[1]
    elif learner == 'random_forest':
        return y_pred, te_data['y'], best_n_estimators, best_criterion, X_te.shape[1]


def __single_experiment(configuration, learner, X_tr, X_te, y_tr, y_te):
    if learner == 'lr':
        C = configuration
        cls = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, C=C)
    elif learner == 'svm':
        C = configuration[0]
        kernel = configuration[1]
        cls = SVC(class_weight='balanced', random_state=42, C=C, kernel=kernel)
    elif learner == 'random_forest':
        n_estimators = configuration[0]
        criterion = configuration[1]
        cls = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=n_estimators, criterion=criterion)
    cls.fit(X_tr, y_tr)
    y_pred = cls.predict(X_te)
    f1 = f1_score(y_te, y_pred, average='macro')
    return f1


# in case of other parameters to optimize, change here
def __create_GridSearch(learner):
    # for svm and lr
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # for svm
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    #for random_forest
    n_estimators = [10, 50, 100, 150]
    criterions = ['gini', 'entropy']
    configurations = []
    if learner in ['svm', 'lr']:
        for C in Cs:
            if learner == 'svm':
                for kernel in kernels:
                    configurations.append((C, kernel))
            else:
                configurations.append(C)
    elif learner == 'random_forest':
        for n_estimator in n_estimators:
            for criterion in criterions:
                configurations.append((n_estimator, criterion))
    return configurations
