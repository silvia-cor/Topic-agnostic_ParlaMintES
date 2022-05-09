from sklearn.model_selection import train_test_split, StratifiedKFold
from rantanplan.core import get_scansion
from multiprocessing import Pool
from functools import partial
import tqdm
from general.utils import pickled_resource, tokenize_nopunct, divide_words_asterisk, find_LCP
from general.LIWC import read_LIWC_dict
from feature_extractor import featuresExtractor
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


# convert the pandas dataframe into a vocabulary of numpy arrays for easy use
def adjust_dataset(dataset, focus):
    if focus == 'party_status':
        labels = dataset['Party_status'].to_numpy()
    else:
        labels = dataset[f'Speaker_{focus}'].to_numpy()
    texts = dataset['Text'].to_numpy()
    pos = dataset['POS-tags'].to_numpy()
    stress = dataset['Stress'].to_numpy()
    liwc_gram = dataset['LIWC_gram'].to_numpy()
    liwc_obj = dataset['LIWC_obj'].to_numpy()
    liwc_cog = dataset['LIWC_cog'].to_numpy()
    liwc_feels = dataset['LIWC_feels'].to_numpy()
    liwc_CDI = dataset['LIWC_CDI'].to_numpy()
    data = {'y': labels, 'texts': texts, 'pos': pos, 'stress': stress, 'liwc_gram': liwc_gram,
            'liwc_obj': liwc_obj, 'liwc_cog': liwc_cog, 'liwc_feels': liwc_feels, 'liwc_CDI': liwc_CDI}
    return data


# divide the dataset into train and test set
# 'ratio' parameters controls the division: 'split' (train_test_split), '50_test' (50 samples per authors as train, the rest as test)
def divide_dataset(dataset, ratio, focus='name'):
    assert ratio in ['split', '50_test'], 'The ratio must be either split or 50_test.'
    if ratio == 'split':
        # divide the dataset into train+val and test
        data = adjust_dataset(dataset, focus)
        trval_texts, te_texts, trval_pos, te_pos, trval_stress, te_stress, trval_liwc_gram, te_liwc_gram, trval_liwc_obj, \
        te_liwc_obj, trval_liwc_cog, te_liwc_cog, trval_liwc_feels, te_liwc_feels, trval_liwc_CDI, te_liwc_CDI, y_trval, y_te = \
            train_test_split(data['texts'], data['pos'], data['stress'], data['liwc_gram'], data['liwc_obj'],
                            data['liwc_cog'], data['liwc_feels'], data['liwc_CDI'], data['y'],
                            test_size=0.1, random_state=42, stratify=data['y'])
        trval_data = {'y': y_trval, 'texts': trval_texts, 'pos': trval_pos, 'stress': trval_stress,
                      'liwc_gram': trval_liwc_gram,
                      'liwc_obj': trval_liwc_obj, 'liwc_cog': trval_liwc_cog, 'liwc_feels': trval_liwc_feels,
                      'liwc_CDI': trval_liwc_CDI}
        te_data = {'y': y_te, 'texts': te_texts, 'pos': te_pos, 'stress': te_stress, 'liwc_gram': te_liwc_gram,
                   'liwc_obj': te_liwc_obj, 'liwc_cog': te_liwc_cog, 'liwc_feels': te_liwc_feels,
                   'liwc_CDI': te_liwc_CDI}
        print(f'#training samples = {len(y_trval)}')
        print(f'#test samples = {len(y_te)}')
    else:
        data_train = dataset.sample(frac=1, random_state=42).groupby('Speaker_name').head(50)  # 50 speeches for each speaker
        data_test = dataset.drop(data_train.index)  # the rest goes in the test set
        trval_data = adjust_dataset(data_train, focus)
        te_data = adjust_dataset(data_test, focus)
        print(f'#training samples = {len(trval_data["y"])}')
        print(f'#test samples = {len(te_data["y"])}')
    return trval_data, te_data


# change the labels for binary classification
# 1 = pos class (author) | 0 = neg class (other authors)
def create_bin_task_labels(data, label):
    label_data = data.copy()
    label_data['y'] = [1 if item == label else 0 for item in data['y']]
    return label_data


# generates the name of the method (used to save the results)
def create_method_name(feats):
    method_name = ''
    if feats['base_features']:
        method_name += 'BaseFeatures'
    if feats['pos_tags']:
        if method_name != '':
            method_name += ' + '
        method_name += 'POS'
    if feats['stress']:
        if method_name != '':
            method_name += ' + '
        method_name += 'STRESS'
    if feats['liwc_gram']:
        if method_name != '':
            method_name += ' + '
        method_name += 'LIWC_GRAM'
    if feats['liwc_obj']:
        if method_name != '':
            method_name += ' + '
        method_name += 'LIWC_OBJ'
    if feats['liwc_cog']:
        if method_name != '':
            method_name += ' + '
        method_name += 'LIWC_COG'
    if feats['liwc_feels']:
        if method_name != '':
            method_name += ' + '
        method_name += 'LIWC_FEELS'
    if method_name == 'BaseFeatures + POS + STRESS + LIWC_GRAM + LIWC_COG + LIWC_FEELS':
        method_name = 'ALL'
    if method_name == 'POS + STRESS + LIWC_GRAM + LIWC_COG + LIWC_FEELS':
        method_name = 'ALL - BaseFeatures'
    if method_name == 'BaseFeatures + STRESS + LIWC_GRAM + LIWC_COG + LIWC_FEELS':
        method_name = 'ALL - POS'
    if method_name == 'BaseFeatures + POS + LIWC_GRAM + LIWC_COG + LIWC_FEELS':
        method_name = 'ALL - STRESS'
    if method_name == 'BaseFeatures + POS + STRESS + LIWC_COG + LIWC_FEELS':
        method_name = 'ALL - LIWC_GRAM'
    if method_name == 'BaseFeatures + POS + STRESS + LIWC_GRAM + LIWC_FEELS':
        method_name = 'ALL - LIWC_COG'
    if method_name == 'BaseFeatures + POS + STRESS + LIWC_GRAM + LIWC_COG':
        method_name = 'ALL - LIWC_FEELS'
    return method_name

# gets the category label for each author
def get_speaker_label(dataset, label_type):
    assert label_type in ['wing', 'party', 'birth', 'gender'], 'Can get labels for wing, party, birth or gender.'
    labels = []
    authors = dataset['Speaker_name'].unique()
    for author in authors:
        author_label = dataset[dataset['Speaker_name'] == author][f'Speaker_{label_type}'].to_list()
        if len(np.unique(author_label)) > 1:
            label = max(set(author_label), key=author_label.count)
            label = ''.join(label)
            labels.append(label)
        else:
            labels.extend(np.unique(author_label))
    if label_type == 'birth':
        labels = [round(int(label) / 10) * 10 for label in labels]
    return labels


# ------------------------------------------------------------------------
# encoding methods
# ------------------------------------------------------------------------

# STRESS ENCODING
def stress_encoding(texts, lang='es'):
    print('Creating stress encodings...')
    with Pool(processes=6) as p:
        if lang == 'es':
            stress_texts = list(tqdm.tqdm(p.imap(_ES_stress, texts), total=len(texts)))
    return stress_texts


# we use the rantanplan library for encoding the stresses
# https://github.com/linhd-postdata/rantanplan
def _ES_stress(text):
    return get_scansion(text)[0]['rhythm']['stress']


# LIWC ENCODING
def LIWC_encoding(texts, cat, lang='es', liwc_path="../pickles/liwc_es.pickle"):
    if lang == 'es':
        assert cat in ['gram', 'obj', 'cog', 'feels', 'CDI'], 'LIWC categories for ES are: gram, obj, cog, feels, CDI'
        dic_liwc = pickled_resource(liwc_path, read_LIWC_dict)
        print(f'Creating LIWC {cat} encoding...')
        with Pool(processes=8) as p:
            if cat == 'gram':
                encoding = partial(_liwc_match, dic=dic_liwc['dic_gram'])
            if cat == 'obj':
                encoding = partial(_liwc_match, dic=dic_liwc['dic_obj'])
            if cat == 'cog':
                encoding = partial(_liwc_match, dic=dic_liwc['dic_cog'])
            if cat == 'feels':
                encoding = partial(_liwc_match, dic=dic_liwc['dic_feels'])
            if cat == 'CDI':
                encoding = partial(_liwc_match, dic=dic_liwc['dic_CDI'])
            liwc_texts = list(tqdm.tqdm(p.imap(encoding, texts), total=len(texts)))
        return liwc_texts


# encode each word in text with the corresponding LIWC category
def _liwc_match(text, dic):
    encoded_text = []
    options_without_ast, options_with_ast = divide_words_asterisk(list(dic.keys()))
    for word in tokenize_nopunct(text):
        match = find_match(word, options_without_ast, options_with_ast)
        encoded_text.append(dic.get(match, "w"))
    encoded_text = ' '.join(encoded_text)
    return encoded_text


# find the longest match in the dictionary
def find_match(word, options_without_ast, options_with_ast):
    if word in options_without_ast:
        return word
    else:
        best_options = [option for option in options_with_ast if word.startswith(option[:-1])]
        if best_options:
            LCPs = [find_LCP(word, best_option) for best_option in best_options]
            return best_options[LCPs.index(max(LCPs))]
        else:
            return ''


# creates feature matrixes for kfold
def create_kfold_matrix(trval_data, feats, split):
    train_idx = split[0]
    val_idx = split[1]
    tr_data = {}
    val_data = {}
    for key in trval_data:
        tr_data[key] = [trval_data[key][idx] for idx in train_idx]
        val_data[key] = [trval_data[key][idx] for idx in val_idx]
    X_tr, X_val, _ = featuresExtractor(tr_data, val_data, **feats)
    return X_tr, X_val, tr_data['y'], val_data['y']


# find and count patterns in authors' speeches
def find_pattern(dataset, pattern, where):
    authors = dataset['Speaker_name'].unique()
    for author in authors:
        df = dataset[dataset['Speaker_name'] == author]
        df = adjust_dataset(df, focus='name')
        n = 0
        for item in df[where]:
            n += item.count(pattern)
        print(f'Count {pattern} in {author}: {n}')
