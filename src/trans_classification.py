import pandas as pd
from general.helpers import divide_dataset, create_bin_task_labels
from simpletransformers.classification import ClassificationModel
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import f1_score
import os
from pathlib import Path
import pickle
from general.sign_test import significance_test

model_args = {"num_train_epochs": 50, "output_dir": "../trans-model", "cache_dir": "../cache_dir",
              'overwrite_output_dir': True, 'cuda_device': 1, 'manual_seed': 42, 'learning_rate': 0.000001}


# classification experiments with BETO transformer (MC) and (BC)
# at the end, comparison with best classic ML method (if the files exist)
def trans_classification(dataset, focus, output_path, pickle_path, task='MC'):
    labels = dataset[f'Speaker_{focus}'].unique()
    if os.path.exists(output_path):
        df_csv = pd.read_csv(output_path, sep=';')
        with open(pickle_path, 'rb') as pickle_file:
            df_preds = pickle.load(pickle_file)
    else:
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
        df_preds = {}
        if task == 'MC':
            df_csv = pd.DataFrame(columns=['Method', 'Macro_F1', 'micro_F1'])
        else:
            columns = ['Method']
            columns.extend(labels)
            df_csv = pd.DataFrame(columns=columns)
    method_name = 'Beto_base_cased'
    if method_name in df_csv['Method'].values:
        print(f'{task} transformer experiment {method_name} already done!')
    else:
        print(f'----- {task} transformer experiment {method_name} -----')
        trval_data, te_data = divide_dataset(dataset, ratio='50_test', focus=focus)
        if task == 'MC':
            df_trval, df_te = _labels_to_int(trval_data, te_data)
            num_labels = len(np.unique(trval_data['y']))
            model = ClassificationModel("bert",
                                        "dccuchile/bert-base-spanish-wwm-cased",
                                        num_labels=num_labels,
                                        args=model_args)
            model.train_model(df_trval)
            preds, raw_outputs = model.predict(df_te['text'].to_list())
            le = preprocessing.LabelEncoder().fit(trval_data['y'])
            df_preds[method_name] = le.inverse_transform(preds)
            macro_f1 = f1_score(df_te['labels'].to_list(), preds, average='macro')
            micro_f1 = f1_score(df_te['labels'].to_list(), preds, average='micro')
            row = {'Method': method_name, 'Macro_F1': np.around(macro_f1, decimals=3), 'micro_F1': np.around(micro_f1, decimals=3)}
            df_csv = df_csv.append(row, ignore_index=True)
            print('----- THE END -----')
            print(f'Macro-F1:', df_csv[df_csv['Method'] == method_name]['Macro_F1'].values[0])
            print(f'micro-F1:', df_csv[df_csv['Method'] == method_name]['micro_F1'].values[0])
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(df_preds, pickle_file)
        else:
            f1s = []
            for label in labels:
                print(f'Experiment for {label}')
                if label not in df_preds:
                    df_preds[label] = {}
                df_tr, df_te = _labels_to_int(trval_data, te_data, label)
                model = ClassificationModel("bert", "dccuchile/bert-base-spanish-wwm-cased", args=model_args)
                model.train_model(df_tr)
                preds, raw_outputs = model.predict(df_te['text'].to_list())
                df_preds[label][method_name] = preds
                f1 = np.around(f1_score(df_te['labels'].to_list(), preds, average='binary'), decimals=3)
                print(f'F1: {f1}\n')
                f1s.append((label, f1))
            row = {'Method': method_name}
            for label, f1 in f1s:
                row[label] = f1
            df_csv = df_csv.append(row, ignore_index=True)
            print('----- THE END -----')
            print(f'F1 per author:')
            for label in labels:
                print(label, df_csv[df_csv['Method'] == method_name][label].values[0])
        df_csv.to_csv(path_or_buf=output_path, sep=';', index=False, header=True)
        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(df_preds, pickle_file)
    # compare best classic ML method and transformer
    ml_csv_path = f'../output/ParlaMint_{task}_svm_{focus}.csv'
    ml_pickle_path = f'../pickles/ParlaMint_{task}_svm_{focus}.pickle'
    assert os.path.exists(ml_csv_path) and os.path.exists(ml_pickle_path), 'No files for classic ML methods'
    ml_df = pd.read_csv(ml_csv_path, sep=';')
    beto_df = pd.read_csv(output_path, sep=';')
    with open(ml_pickle_path, 'rb') as pickle_file:
        ml_preds = pickle.load(pickle_file)
    with open(pickle_path, 'rb') as pickle_file:
        beto_preds = pickle.load(pickle_file)
    if task == 'MC':
        best_mls = ml_df[ml_df['Macro_F1'].values == ml_df['Macro_F1'].values.max()]['Method'].values
        best_betos = beto_df[beto_df['Macro_F1'].values == beto_df['Macro_F1'].values.max()]['Method'].values
        print('The best features methods are:', best_mls)
        print('The best BETO methods are:', best_betos)
        for best_beto in best_betos:
            for best_ml in best_mls:
                print(f'COMPARISON {best_ml} vs {best_beto} (baseline)')
                ml_macro = ml_df[ml_df['Method'] == best_ml]['Macro_F1'].values[0]
                beto_macro = beto_df[beto_df['Method'] == best_beto]['Macro_F1'].values[0]
                delta_macro = (ml_macro - beto_macro) / beto_macro * 100
                ml_micro = ml_df[ml_df['Method'] == best_ml]['micro_F1'].values[0]
                beto_micro = beto_df[beto_df['Method'] == best_beto]['micro_F1'].values[0]
                delta_micro = (ml_micro - beto_micro) / beto_micro * 100
                print(f'Macro-F1 Delta %: {delta_macro:.2f}')
                print(f'Micro-F1 Delta %: {delta_micro:.2f}')
                significance_test(ml_preds['True'], beto_preds[best_beto], ml_preds[best_ml], 'Beto')
    else:
        if focus == 'party_status':
            unique_labels = dataset['Party_status'].unique()
        else:
            unique_labels = dataset[f'Speaker_{focus}'].unique()
        for label in unique_labels:
            print(f'Comparison for {label}')
            best_methods = ml_df[ml_df[label].values == ml_df[label].values.max()]['Method'].values
            best_betos = beto_df[beto_df[label].values == beto_df[label].values.max()]['Method'].values
            print('The best features methods are:', best_methods)
            print('The best BETO methods are:', best_betos)
            for best_beto in best_betos:
                for best_method in best_methods:
                    print(f'COMPARISON {best_method} vs {best_beto} (baseline)')
                    ml_f1 = ml_df[ml_df['Method'] == best_method][label].values[0]
                    beto_f1 = beto_df[beto_df['Method'] == best_beto][label].values[0]
                    delta_f1 = (ml_f1 - beto_f1) / beto_f1 * 100
                    print(f'F1 Delta %: {delta_f1:.2f}')
                    significance_test(ml_preds[label]['True'], beto_preds[label][best_beto], ml_preds[label][best_method], 'Beto')


# change categories labels into int labels
def _labels_to_int(trval_data, te_data, label=None):
    if label is None:
        le = preprocessing.LabelEncoder().fit(trval_data['y'])
        trval = {'text': trval_data['texts'], 'labels': le.transform(trval_data['y'])}
        te = {'text': te_data['texts'], 'labels': le.transform(te_data['y'])}
    else:
        trval_data = create_bin_task_labels(trval_data, label)
        te_data = create_bin_task_labels(te_data, label)
        trval = {'text': trval_data['texts'], 'labels': trval_data['y']}
        te = {'text': te_data['texts'], 'labels': te_data['y']}
    df_tr = pd.DataFrame(trval)
    df_te = pd.DataFrame(te)
    return df_tr, df_te


def trans_party_status_experiment(dataset, output_path, pickle_path, author='Sánchez Pérez-castejón, Pedro'):
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
    method_name = 'Beto_base_cased'
    if method_name in df_csv['Method'].values:
        print(f'BC transformer experiment {method_name} already done!')
    else:
        f1s = []
        for status in statuses:
            print(f'---- BC transformer experiment {method_name} ----')
            print(f'--- Party-status experiment for {author} when in {status} ---')
            if status not in df_preds:
                df_preds[status] = {}
            mini_dataset = dataset.drop(dataset[(dataset.Speaker_name == author) & (dataset.Party_status != status)].index)
            trval_data, te_data = divide_dataset(mini_dataset, ratio='50_test', focus='name')
            df_tr, df_te = _labels_to_int(trval_data, te_data, author)
            model = ClassificationModel("bert", "dccuchile/bert-base-spanish-wwm-cased", args=model_args)
            model.train_model(df_tr)
            preds, raw_outputs = model.predict(df_te['text'].to_list())
            df_preds[status][method_name] = preds
            f1 = np.around(f1_score(df_te['labels'].to_list(), preds, average='binary'), decimals=3)
            print(f'F1: {f1}\n')
            f1s.append((status, f1))
        row = {'Method': method_name}
        for status, f1 in f1s:
            row[status] = f1
        df_csv = df_csv.append(row, ignore_index=True)
    df_csv.to_csv(path_or_buf=output_path, sep=';', index=False, header=True)
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(df_preds, pickle_file)
