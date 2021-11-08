from sklearn.svm import SVC
from sklearn.metrics import f1_score
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from feature_extractor import featuresExtractor
from general.sign_test import significance_test
from general.utils import divide_dataset, create_method_name
import tqdm
from multiprocessing import Pool
from functools import partial

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


# ------------------------------------------------------------------------
# method to do (classical ML) classification
# ------------------------------------------------------------------------

def build_SVM_experiment(dataset, feats, output_path):
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
    else:
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        df = {}
    method_name = create_method_name(feats)
    if method_name in df:
        print(f'Experiment {method_name} already done!')
    else:
        print(f'----- SVM EXPERIMENT {method_name} -----')
        tr_data, val_data, te_data = divide_dataset(dataset)
        y_pred, y_te, best_C, best_kernel, tot_features = _GridSeach_classification(tr_data, val_data, te_data, feats)
        if 'True' not in df:
            df['True'] = {}
            df['True']['labels'] = y_te
        df[method_name] = {}
        df[method_name]['preds'] = y_pred
        df[method_name]['best_C'] = best_C
        df[method_name]['best_kernel'] = best_kernel
        df[method_name]['tot_features'] = tot_features
        macro_f1 = f1_score(df['True']['labels'], df[method_name]['preds'], average='macro')
        micro_f1 = f1_score(df['True']['labels'], df[method_name]['preds'], average='micro')
        df[method_name]['macroF1'] = macro_f1
        df[method_name]['microF1'] = micro_f1
    print('----- THE END -----')
    print('Tot. features:', int(np.round(df[method_name]['tot_features'])))
    print(f'Macro-F1: {df[method_name]["macroF1"]:.3f}')
    print(f'Micro-F1: {df[method_name]["microF1"]:.3f}')
    df.to_csv(path_or_buf=output_path)
    # significance test if SQ are in the features with another method
    # significance test is against the same method without SQ
    for other_method in df:
        if len(df) == 1 and df[other_method] == method_name:
            print(f'{method_name} is the only method stored, no significance test is performed')
        else:
            if other_method != method_name:
                print(f'COMPARISON WITH BASELINE {other_method}')
                delta_macro = (df[method_name]['macroF1'] - df[other_method]['macroF1']) / df[other_method]['macroF1'] * 100
                delta_micro = (df[method_name]['microF1'] - df[other_method]['microF1']) / df[other_method]['microF1'] * 100
                print(f'Macro-F1 Delta %: {delta_macro:.2f}')
                print(f'Micro-F1 Delta %: {delta_micro:.2f}')
                significance_test(df['True']['labels'], df[other_method]['preds'], df[method_name]['preds'], other_method)


# perform train-val-test validation using a SVM
# optimization of parameters via GridSearch from scratch
def _GridSeach_classification(tr_data, val_data, te_data, feats):
    # create GridSearch
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    configurations = []
    # in case of other parameters to optimize, add here
    for C in Cs:
        for kernel in kernels:
            configurations.append((C, kernel))
    print('PARAMETERS OPTIMIZATION')
    with Pool(processes=4) as p:
        optimization_step = partial(__single_experiment, feats=feats, tr_data=tr_data, val_data=val_data)
        results = list(tqdm.tqdm(p.imap(optimization_step, configurations), total=len(configurations)))
    #results = []
    #for configuration in configurations:
    #    results.append(__single_experiment(configuration, feats, tr_data, val_data))
    best_result_idx = results.index(max(results, key=lambda result: result))
    best_C = configurations[best_result_idx][0]
    best_kernel = configurations[best_result_idx][1]
    print('BEST MODEL')
    print('Best C:', best_C)
    print('Best Kernel:', best_kernel)
    print(f'Best macro-f1: {results[best_result_idx]:.3f}')
    print('CLASSIFICATION')
    cls = SVC(class_weight='balanced', random_state=42, C=best_C)
    trval_data = {key: tr_data[key] + val_data[key] for key in tr_data}
    X_trval, X_te = featuresExtractor(trval_data, te_data, **feats)
    cls.fit(X_trval, trval_data['y'])
    y_pred = cls.predict(X_te)
    print("Training shape: ", X_trval.shape)
    print("Test shape: ", X_te.shape)
    return y_pred, te_data['y'], best_C, best_kernel, X_te.shape[1]


def __single_experiment(configuration, feats, tr_data, val_data):
    C = configuration[0]
    kernel = configuration[1]
    cls = SVC(class_weight='balanced', random_state=42, C=C, kernel=kernel)
    X_tr, X_val = featuresExtractor(tr_data, val_data, **feats)
    cls.fit(X_tr, tr_data['y'])
    y_pred = cls.predict(X_val)
    f1 = f1_score(val_data['y'], y_pred, average='macro')
    return f1
