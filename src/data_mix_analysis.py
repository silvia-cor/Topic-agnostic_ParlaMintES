import pandas as pd
import os
import re
import numpy as np
from conllu import parse_incr
from general.helpers import stress_encoding, LIWC_encoding


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


def process_ParlaMint(data_path='../dataset/ParlaMint-ES.conllu', file_path='/ParlaMint-ES.csv'):
    print('--- CREATING DATASET PICKLE ---')
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

    df = df[df.groupby(field).Speaker_name.transform('count') >= 150]  # take Speakers with >=150 entries
    df.loc[(df['Speaker_name'] == 'Martínez Seijo, María Luz') & (df[field] == 'PP;PSOE;UP'), field] = 'PSOE'
    df.loc[(df[field] == 'PP;PSOE') & (df['Speaker_name'].isin(['González Veracruz, María',
                                                                        'Martínez Seijo, María Luz',
                                                                        'Martín González, María Guadalupe'])), field] = 'PSOE'
    df = df.drop(df[(df[field] == 'PP;PSOE') & (df['Speaker_name'] == 'Rodríguez Ramos, María Soraya')].index)
    # re-label the parties
    df.loc[(df[field].isin(['PSOE', 'PSC-PSOE'])), field] = 'Izquierda'
    df.loc[(df[field].isin(['PP', 'PP-Foro'])), field] = 'Derecha'
    df.loc[(df[field].isin(['ERC-S', 'EH Bildu', 'ERC-CATSÍ', 'UP'])), field] = 'Más izquierda'
    df.loc[(df[field].isin(['Vox'])), field] = 'Más derecha'
    df.loc[(df[field].isin(['EAJ-PNV', 'JxCat-Junts', 'CDC', 'CiU'])), field] = 'Regionalistas'
    df.loc[(df[field].isin(['Cs'])), field] = 'Centro'
    df = df[df['Text'].notna()]
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
    stress_texts = stress_encoding(texts)
    print('Creating LIWC encodings...')
    liwc_gram_texts = LIWC_encoding(texts, cat='gram')
    liwc_obj_texts = LIWC_encoding(texts, cat='obj')
    liwc_cog_texts = LIWC_encoding(texts, cat='cog')
    liwc_feels_texts = LIWC_encoding(texts, cat='feels')
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

def _clean_text_es(text):
    # text = re.sub(r'\[\[(?:(?!\[|\])[\s\S])*\]\]', '', text)
    text = re.sub(r'ä', 'a', text)
    text = re.sub(r'ï', 'i', text)
    text = re.sub(r'ö', 'o', text)
    text = re.sub(r'¿', '¿ ', text)
    text = re.sub(r'¡', '¡ ', text)
    text = text.lower()
    return text