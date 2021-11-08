import pandas as pd
import os
import re
import numpy as np
from general.utils import tokenize_nopunct
from rantanplan.core import get_scansion
import tqdm
from multiprocessing import Pool
from functools import partial
from conllu import parse, parse_incr
from general.utils import pickled_resource, find_LCP
from general.LIWC import read_LIWC_dict


# create a single .csv file from all the files in the dataset ParlaMint (conllu version)
# the conllu version has already the POS-tags (and much more)
def create_ParlaMint_csv(data_path='../dataset/ParlaMint-ES.conllu', final_file_path='/ParlaMint-ES.csv', lang='es'):
    print('Creating the PARLAMINT csv file...')
    frames = []
    for file_name in os.listdir(data_path):
        file_path = data_path + '/' + file_name
        if file_name.endswith(".tsv"):
            file = open(file_path)
            frame = pd.read_csv(file, sep='\t')
            frames.append(frame)
    df = pd.concat(frames)
    df["Text"] = ''
    df["POS-tags"] = ''
    print('Tot. files to process:', len(os.listdir(data_path)))
    for file_name in os.listdir(data_path):
        file_path = data_path + '/' + file_name
        if file_name.endswith(".conllu") and file_name != "00README.txt":
            with open(file_path) as f:
                for item in parse_incr(f):
                    text = item.metadata['text']
                    if not text.isupper():  # all-uppercase texts are questions from the public
                        id_split = re.split('(\.u[0-9]*)', item.metadata['sent_id'])
                        id = id_split[0] + id_split[1]
                        if lang == 'es':
                            text = _clean_text_es(text)
                        pos_text = ' '.join(token['upos'] for token in item)
                        if df['Text'][df['ID'] == id].iloc[0] != '':
                            df.loc[df.ID == id, "Text"] += ' '
                            df.loc[df.ID == id, "POS-tags"] += ' '
                        df.loc[df.ID == id, "Text"] += text
                        df.loc[df.ID == id, "POS-tags"] += pos_text
    df.to_csv(data_path + final_file_path, index=False, sep='\t')


# TODO: add Italian option
# TODO: add LIWC
# analyze the .csv file of ParlaMint to extract some statistics
def process_ParlaMint(data_path='../dataset/ParlaMint-ES.conllu', file_path='/ParlaMint-ES.csv', focus='name',
                      lang='es'):
    print('--- CREATING DATASET PICKLE ---')
    assert focus in ['name', 'party', 'gender'], 'The project focus must be in either name, party or gender'
    assert lang in ['es', 'it'], 'The project language must be either es (spanish) or it (italian).'
    if not os.path.isfile(data_path + file_path):
        create_ParlaMint_csv(data_path)
    else:
        print('PARLAMINT csv file found')
    df = pd.read_csv(data_path + file_path, sep='\t')  # read csv file
    df = df[df.Speaker_role != 'Chairperson']  # delete ChairPerson entries
    texts = []
    labels = []
    unique_labels = []
    labels_lens = []
    pos_tags_texts = []
    field = 'Speaker_' + focus
    if lang == 'es':
        if focus == 'name':
            df = df[df.groupby(field).Speaker_name.transform('count') >= 150]  # take Speakers with >=100 entries
            unique_labels = df[field].unique()
            print('Unique speakers:', len(unique_labels))
        elif focus == 'party':
            df = df[df[field].notna()]  # delete entries with NaN values
            # cleaning the dataset from 'difficult' entries
            df.loc[(df['Speaker_name'] == 'Martínez Seijo, María Luz') & (df[field] == 'PP;PSOE;UP'), field] = 'PSOE'
            df.loc[(df[field] == 'PP;PSOE') & (df['Speaker_name'].isin(['González Veracruz, María',
                                                                        'Martínez Seijo, María Luz',
                                                                        'Martín González, María Guadalupe'])), field] = 'PSOE'
            df = df.drop(df[(df[field] == 'PP;PSOE') & (df['Speaker_name'] == 'Rodríguez Ramos, María Soraya')].index)
            df = df[df.groupby(field).Speaker_party.transform('count') >= 300]  # take Parties with >=300 entries
            unique_labels = df[field].unique()
            print('Unique parties:', len(unique_labels), unique_labels)
        elif focus == 'gender':
            unique_labels = df[field].unique()
    print('Dataset shape:', df.shape)
    #  create the categorical labels
    for i, label in enumerate(unique_labels):
        mini_df = df.loc[df[field] == label]['Text']
        labels_lens.append(sum(len(text) for text in mini_df))
        texts.extend(mini_df)
        labels.extend([i] * len(mini_df))
        pos_tags_texts.extend(df.loc[df[field] == label]['POS-tags'])
    print('Minimum amount of text:', unique_labels[labels_lens.index(min(labels_lens))], min(labels_lens))
    print('Maximum amount of text:', unique_labels[labels_lens.index(max(labels_lens))], max(labels_lens))
    print(f'Mean among classes: {np.mean(labels_lens):.2f}')
    print('Creating stress encodings...')
    with Pool(processes=6) as p:
        stress_texts = list(tqdm.tqdm(p.imap(_stress_encoding, texts), total=len(texts)))
    print('Creating LIWC encodings...')
    liwc_path = f"../pickles/liwc_es.pickle"
    dic_liwc = pickled_resource(liwc_path, read_LIWC_dict)
    print('LIWC encodings:', dic_liwc.keys())
    with Pool(processes=6) as p:
        encode_gram = partial(_liwc_encode, dic=dic_liwc['dic_gram'])
        liwc_gram_texts = list(tqdm.tqdm(p.imap(encode_gram, texts), total=len(texts)))
    with Pool(processes=6) as p:
        encode_obj = partial(_liwc_encode, dic=dic_liwc['dic_obj'])
        liwc_obj_texts = list(tqdm.tqdm(p.imap(encode_obj, texts), total=len(texts)))
    with Pool(processes=6) as p:
        encode_cog = partial(_liwc_encode, dic=dic_liwc['dic_cog'])
        liwc_cog_texts = list(tqdm.tqdm(p.imap(encode_cog, texts), total=len(texts)))
    with Pool(processes=6) as p:
        encode_feels = partial(_liwc_encode, dic=dic_liwc['dic_feels'])
        liwc_feels_texts = list(tqdm.tqdm(p.imap(encode_feels, texts), total=len(texts)))
    dataset = {'unique_labels': np.array(unique_labels),
               'labels': np.array(labels),
               'texts': np.array(texts),
               'pos_tags_texts': np.array(pos_tags_texts),
               'stress_texts': np.array(stress_texts),
               'liwc_gram_texts': np.array(liwc_gram_texts),
               'liwc_obj_texts': np.array(liwc_obj_texts),
               'liwc_cog_texts': np.array(liwc_cog_texts),
               'liwc_feels_texts': np.array(liwc_feels_texts)
               }
    return dataset


# we use the rantanplan library for encoding the stresses
# https://github.com/linhd-postdata/rantanplan
# TODO: also for italian
def _stress_encoding(text):
    return get_scansion(text)[0]['rhythm']['stress']


def _clean_text_es(text):
    # text = re.sub(r'\[\[(?:(?!\[|\])[\s\S])*\]\]', '', text)
    text = re.sub(r'ä', 'a', text)
    text = re.sub(r'ï', 'i', text)
    text = re.sub(r'ö', 'o', text)
    text = re.sub(r'¿', '¿ ', text)
    text = re.sub(r'¡', '¡ ', text)
    text = text.lower()
    return text


def _liwc_encode(text, dic):
    encoded_text = []
    for word in tokenize_nopunct(text):
        match = __find_match(word, list(dic.keys()))
        encoded_text.append(dic[match])
    encoded_text = ' '.join(encoded_text)
    return encoded_text


# find the longest match in the dictionary
def __find_match(word, options):
    options = sorted(options)
    best_option = ''
    longest_prefix = 0
    for option in options:
        if word == option:
            return option
        else:
            lcp = find_LCP(word, option)
            if lcp > longest_prefix:
                longest_prefix = lcp
                best_option = option
    return best_option

