import os
import pickle
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


# pickle the output of a function
def pickled_resource(pickle_path: str, generation_func: callable, *args, **kwargs):
    if pickle_path is None:
        return generation_func(*args, **kwargs)
    else:
        if os.path.exists(pickle_path):
            return pickle.load(open(pickle_path, 'rb'))
        else:
            instance = generation_func(*args, **kwargs)
            os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
            pickle.dump(instance, open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            return instance


# tokenize text without punctuation
def tokenize_nopunct(text):
    unmod_tokens = nltk.word_tokenize(text)
    return [token.lower() for token in unmod_tokens if
            any(char.isalpha() for char in token)]  # checks whether all the chars are alphabetic


# return list of function words
def get_function_words(lang):
    languages = {'es': 'spanish',
                 'it': 'italian'
                 }
    if lang not in languages:
        raise ValueError('{} not in scope!'.format(lang))
    else:
        return stopwords.words(languages[lang])


# find the Longest Common Prefix among two words (value)
# if the second word is longer than the first word, return 0
# in case the LCP is less than 2/3 of the word, return 0
def find_LCP(x, y):
    prefix_length = 0
    word_por = (len(x)//3)*2
    if len(y)-1 > len(x):
        return 0
    for i in range(0, min(len(x), len(y))):
        if x[i] == y[i]:
            prefix_length += 1
        else:
            break
    if prefix_length >= word_por:
        return prefix_length
    else:
        return 0


# divide the dataset for train-val-test experiment
def divide_dataset(dataset):
    # divide the dataset into train+val and test
    trval_texts, te_texts, trval_pos, te_pos, trval_stress, te_stress, \
    trval_liwc_gram, te_liwc_gram, trval_liwc_obj, te_liwc_obj, trval_liwc_cog, te_liwc_cog, trval_liwc_feels, te_liwc_feels, \
    y_trval, y_te = train_test_split(dataset['texts'], dataset['pos_tags_texts'], dataset['stress_texts'],
                                     dataset['liwc_gram_texts'], dataset['liwc_obj_texts'], dataset['liwc_cog_texts'],
                                     dataset['liwc_feels_texts'],
                                     dataset['labels'], test_size=0.1, random_state=42, stratify=dataset['labels'])
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
