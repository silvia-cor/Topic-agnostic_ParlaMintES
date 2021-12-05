from sklearn.model_selection import train_test_split
from rantanplan.core import get_scansion
from multiprocessing import Pool
from functools import partial
import tqdm
from general.utils import pickled_resource, tokenize_nopunct, divide_words_asterisk, find_LCP
from general.LIWC import read_LIWC_dict
import pandas as pd


# divide the dataset for train-val-test experiment
def divide_dataset(dataset):
    labels = dataset['Speaker_name'].to_numpy()
    texts = dataset['Text'].to_numpy()
    pos = dataset['POS-tags'].to_numpy()
    stress = dataset['Stress'].to_numpy()
    liwc_gram = dataset['LIWC_gram'].to_numpy()
    liwc_obj = dataset['LIWC_obj'].to_numpy()
    liwc_cog = dataset['LIWC_cog'].to_numpy()
    liwc_feels = dataset['LIWC_feels'].to_numpy()
    # divide the dataset into train+val and test
    trval_texts, te_texts, trval_pos, te_pos, trval_stress, te_stress, \
    trval_liwc_gram, te_liwc_gram, trval_liwc_obj, te_liwc_obj, trval_liwc_cog, te_liwc_cog, trval_liwc_feels, te_liwc_feels, \
    y_trval, y_te = train_test_split(texts, pos, stress, liwc_gram, liwc_obj, liwc_cog, liwc_feels, labels, test_size=0.1, random_state=42, stratify=labels)
    # divide the train+val so that the dataset is train/val/test
    tr_texts, val_texts, tr_pos, val_pos, tr_stress, val_stress, \
    tr_liwc_gram, val_liwc_gram, tr_liwc_obj, val_liwc_obj, tr_liwc_cog, val_liwc_cog, tr_liwc_feels, val_liwc_feels, \
    y_tr, y_val = train_test_split(trval_texts, trval_pos, trval_stress,
                                   trval_liwc_gram, trval_liwc_obj, trval_liwc_cog, trval_liwc_feels,
                                   y_trval, test_size=0.1, random_state=42, stratify=y_trval)
    print(f'#training samples = {len(y_tr)}')
    print(f'#validation samples = {len(y_val)}')
    print(f'#test samples = {len(y_te)}')

    tr_data = {'y': y_tr, 'texts': tr_texts, 'pos': tr_pos, 'stress': tr_stress, 'liwc_gram': tr_liwc_gram,
               'liwc_obj': tr_liwc_obj, 'liwc_cog': tr_liwc_cog, 'liwc_feels': tr_liwc_feels}
    val_data = {'y': y_val, 'texts': val_texts, 'pos': val_pos, 'stress': val_stress, 'liwc_gram': val_liwc_gram,
                'liwc_obj': val_liwc_obj, 'liwc_cog': val_liwc_cog, 'liwc_feels': val_liwc_feels}
    te_data = {'y': y_te, 'texts': te_texts, 'pos': te_pos, 'stress': te_stress, 'liwc_gram': te_liwc_gram,
               'liwc_obj': te_liwc_obj, 'liwc_cog': te_liwc_cog, 'liwc_feels': te_liwc_feels}
    return tr_data, val_data, te_data


# generates the name of the method used to save the results
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
    return method_name


# ------------------------------------------------------------------------
# encoding methods
# ------------------------------------------------------------------------

# STRESS ENCODING
# TODO: also for italian
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
def LIWC_encoding(texts, cat, lang='es'):
    if lang == 'es':
        liwc_path = f"../pickles/liwc_es.pickle"
        assert cat in ['gram', 'obj', 'cog', 'feels'], 'LIWC categories for ES are: gram, obj, cog, feels'
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
            liwc_texts = list(tqdm.tqdm(p.imap(encoding, texts), total=len(texts)))
        return liwc_texts


# encode each word in text with the corresponding LIWC category
def _liwc_match(text, dic):
    encoded_text = []
    options_without_ast, options_with_ast = divide_words_asterisk(list(dic.keys()))
    for word in tokenize_nopunct(text):
        match = find_match(word, options_without_ast, options_with_ast)
        encoded_text.append(dic.get(match, "0"))
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
