import pandas as pd
import os
import re
import numpy as np
from conllu import parse_incr
from general.helpers import stress_encoding, LIWC_encoding


# create a single .csv file from all the files in the dataset ParlaMint (conllu version)
# the conllu version has already the POS-tags (and much more)
def create_ParlaMint_csv(data_path='../dataset/ParlaMint-ES.conllu', final_file_path='/ParlaMint-ES.csv'):
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
                        text = _clean_text_es(text)
                        pos_text = ' '.join(token['upos'] for token in item)
                        if df['Text'][df['ID'] == id].iloc[0] != '':
                            df.loc[df.ID == id, "Text"] += ' '
                            df.loc[df.ID == id, "POS-tags"] += ' '
                        df.loc[df.ID == id, "Text"] += text
                        df.loc[df.ID == id, "POS-tags"] += pos_text
    df.to_csv(data_path + final_file_path, index=False, sep='\t')


# DATASET WITH 5 SPEAKERS PER WING (6 WINGS) -> SAMPLES UNBALANCED
def process_ParlaMint_6wings(data_path='../dataset/ParlaMint-ES.conllu', file_path='/ParlaMint-ES.csv', liwc_path="../pickles/liwc_es.pickle"):
    print('--- CREATING DATASET PICKLE ---')
    # assert focus in ['name', 'party'], 'The project focus must be either name or party.'
    if not os.path.isfile(data_path + file_path):
        create_ParlaMint_csv(data_path)
    else:
        print('PARLAMINT csv file found')
    df = pd.read_csv(data_path + file_path, sep='\t')  # read csv file
    df = df[df.Speaker_role != 'Chairperson']  # delete ChairPerson entries
    # df = df.groupby("Speaker_name").filter(lambda x: len(x) >= 150)
    # resolve ambiguous Speaker_party entries
    # df.loc[(df['Speaker_name'] == 'Martínez Seijo, María Luz') & (df['Speaker_party'] == 'PP;PSOE;UP'), 'Speaker_party'] = 'PSOE'
    # df.loc[(df['Speaker_party'] == 'PP;PSOE') & (df['Speaker_name'].isin(['González Veracruz, María', 'Martínez Seijo, María Luz', 'Martín González, María Guadalupe'])), 'Speaker_party'] = 'PSOE'
    # df = df.drop(df[(df['Speaker_party'] == 'PP;PSOE') & (df['Speaker_name'] == 'Rodríguez Ramos, María Soraya')].index)
    df = df.groupby("Speaker_party").filter(lambda x: len(x) >= 300)
    # assign Speaker_wing
    df["Speaker_wing"] = np.nan
    df.loc[(df['Speaker_party'].isin(['PSOE', 'PSC-PSOE'])), 'Speaker_wing'] = 'Izquierda'
    df.loc[(df['Speaker_party'].isin(['PP', 'PP-Foro'])), 'Speaker_wing'] = 'Derecha'
    df.loc[(df['Speaker_party'].isin(['ERC-S', 'EH Bildu', 'ERC-CATSÍ', 'UP'])), 'Speaker_wing'] = 'Más izquierda'
    df.loc[(df['Speaker_party'].isin(['Vox'])), 'Speaker_wing'] = 'Más derecha'
    df.loc[(df['Speaker_party'].isin(['EAJ-PNV', 'JxCat-Junts', 'CDC', 'CiU'])), 'Speaker_wing'] = 'Regionalistas'
    df.loc[(df['Speaker_party'].isin(['Cs'])), 'Speaker_wing'] = 'Centro'
    party_groups = df.groupby('Speaker_wing')
    selected_speakers = []
    for name, group in party_groups:
        selected_speakers.extend(group['Speaker_name'].value_counts()[:5].index.tolist())
    df = df[df['Speaker_name'].isin(selected_speakers)]
    df['Text'] = df['Text'].apply(_clean_text_es)  # clean texts
    df = df[df['Text'].notna()]
    texts = df["Text"].to_numpy()
    # creating stress encoding
    df['Stress'] = stress_encoding(texts)
    print('Creating LIWC encodings...')
    df['LIWC_gram'] = LIWC_encoding(texts, cat='gram', liwc_path=liwc_path)
    df['LIWC_obj'] = LIWC_encoding(texts, cat='obj', liwc_path=liwc_path)
    df['LIWC_cog'] = LIWC_encoding(texts, cat='cog', liwc_path=liwc_path)
    df['LIWC_feels'] = LIWC_encoding(texts, cat='feels', liwc_path=liwc_path)
    df['LIWC_CDI'] = LIWC_encoding(texts, cat='CDI', liwc_path=liwc_path)
    return df


# DATASET WITH 5 SPEAKERS PER WING (IZQUIERDA / DERECHA) -> TOO FEW FOR CLUSTERING
def process_ParlaMint_2wings(data_path='../dataset/ParlaMint-ES.conllu', file_path='/ParlaMint-ES.csv', liwc_path="../pickles/liwc_es.pickle"):
    print('--- CREATING DATASET PICKLE ---')
    if not os.path.isfile(data_path + file_path):
        create_ParlaMint_csv(data_path)
    else:
        print('PARLAMINT csv file found')
    df = pd.read_csv(data_path + file_path, sep='\t')  # read csv file
    df = df[df.Speaker_role != 'Chairperson']  # delete ChairPerson entries
    df = df.groupby("Speaker_party").filter(lambda x: len(x) >= 500)
    # assign Speaker_wing
    df["Speaker_wing"] = np.nan
    df.loc[(df['Speaker_party'].isin(['PSOE', 'PSC-PSOE', 'UP'])), 'Speaker_wing'] = 'Izquierda'
    df.loc[(df['Speaker_party'].isin(['PP', 'PP-Foro', 'Vox'])), 'Speaker_wing'] = 'Derecha'
    df.loc[(df['Speaker_party'].isin(['EAJ-PNV', 'JxCat-Junts', 'CDC', 'CiU'])), 'Speaker_wing'] = 'Regionalistas'
    df.loc[(df['Speaker_party'].isin(['Cs'])), 'Speaker_wing'] = 'Centro'
    df = df[df['Speaker_wing'].isin(['Izquierda', 'Derecha'])]
    party_groups = df.groupby('Speaker_wing')
    selected_speakers = []
    for name, group in party_groups:
        selected_speakers.extend(group['Speaker_name'].value_counts()[:5].index.tolist())
    df = df[df['Speaker_name'].isin(selected_speakers)]
    df['Text'].apply(_clean_text_es)  # clean texts
    authors = df['Speaker_name'].unique()
    n_samples = []
    for author in authors:
        n_samples.append(df[df.Speaker_name == author].shape[0])
    df = df.sample(frac=1, random_state=42).groupby('Speaker_name').head(min(n_samples))
    df = df[df['Text'].notna()]
    texts = df["Text"].to_numpy()
    # creating stress encoding
    df['Stress'] = stress_encoding(texts)
    print('Creating LIWC encodings...')
    df['LIWC_gram'] = LIWC_encoding(texts, cat='gram', liwc_path=liwc_path)
    df['LIWC_obj'] = LIWC_encoding(texts, cat='obj', liwc_path=liwc_path)
    df['LIWC_cog'] = LIWC_encoding(texts, cat='cog', liwc_path=liwc_path)
    df['LIWC_feels'] = LIWC_encoding(texts, cat='feels', liwc_path=liwc_path)
    df['LIWC_CDI'] = LIWC_encoding(texts, cat='CDI', liwc_path=liwc_path)
    return df


# DATASET WITH 5 SPEAKERS PER WING (IZQUIERDA / DERECHA / CENTRO / REGIONALISTAS)
def process_ParlaMint_4wings(data_path='../dataset/ParlaMint-ES.conllu', file_path='/ParlaMint-ES.csv', liwc_path="../pickles/liwc_es.pickle"):
    print('--- CREATING DATASET PICKLE ---')
    if not os.path.isfile(data_path + file_path):
        create_ParlaMint_csv(data_path)
    else:
        print('PARLAMINT csv file found')
    df = pd.read_csv(data_path + file_path, sep='\t')  # read csv file
    df = df[df.Speaker_role != 'Chairperson']  # delete ChairPerson entries
    df = df.groupby("Speaker_party").filter(lambda x: len(x) >= 300)
    df = df[df['Text'].str.split().str.len().gt(50)]  # delete speeches with less than 50 words
    # assign Speaker_wing
    df["Speaker_wing"] = np.nan
    df.loc[(df['Speaker_party'].isin(['PSOE', 'PSC-PSOE', 'UP'])), 'Speaker_wing'] = 'Izquierda'
    df.loc[(df['Speaker_party'].isin(['PP', 'PP-Foro', 'Vox'])), 'Speaker_wing'] = 'Derecha'
    df.loc[(df['Speaker_party'].isin(['EAJ-PNV', 'JxCat-Junts', 'CDC', 'CiU'])), 'Speaker_wing'] = 'Regionalista'
    df.loc[(df['Speaker_party'].isin(['Cs'])), 'Speaker_wing'] = 'Centro'
    party_groups = df.groupby('Speaker_wing')
    selected_speakers = []
    for name, group in party_groups:
        selected_speakers.extend(group['Speaker_name'].value_counts()[:5].index.tolist())
    df = df[df['Speaker_name'].isin(selected_speakers)]
    df['Text'].apply(_clean_text_es)  # clean texts
    df = df[df['Text'].notna()]
    texts = df["Text"].to_numpy()
    # creating stress encoding
    df['Stress'] = stress_encoding(texts)
    print('Creating LIWC encodings...')
    df['LIWC_gram'] = LIWC_encoding(texts, cat='gram', liwc_path=liwc_path)
    df['LIWC_obj'] = LIWC_encoding(texts, cat='obj', liwc_path=liwc_path)
    df['LIWC_cog'] = LIWC_encoding(texts, cat='cog', liwc_path=liwc_path)
    df['LIWC_feels'] = LIWC_encoding(texts, cat='feels', liwc_path=liwc_path)
    df['LIWC_CDI'] = LIWC_encoding(texts, cat='CDI', liwc_path=liwc_path)
    return df


def process_ParlaMint_nowing(data_path='../dataset/ParlaMint-ES.conllu', file_path='/ParlaMint-ES.csv', liwc_path="../pickles/liwc_es.pickle"):
    print('--- CREATING DATASET PICKLE ---')
    if not os.path.isfile(data_path + file_path):
        create_ParlaMint_csv(data_path)
    else:
        print('PARLAMINT csv file found')
    df = pd.read_csv(data_path + file_path, sep='\t')  # read csv file
    df = df[df.Speaker_role != 'Chairperson']  # delete ChairPerson entries
    df = df.groupby("Speaker_party").filter(lambda x: len(x) >= 300)
    # assign Speaker_wing
    df["Speaker_wing"] = np.nan
    df.loc[(df['Speaker_party'].isin(['PSOE', 'PSC-PSOE'])), 'Speaker_wing'] = 'Izquierda'
    df.loc[(df['Speaker_party'].isin(['PP', 'PP-Foro'])), 'Speaker_wing'] = 'Derecha'
    df.loc[(df['Speaker_party'].isin(['ERC-S', 'EH Bildu', 'ERC-CATSÍ', 'UP'])), 'Speaker_wing'] = 'Más izquierda'
    df.loc[(df['Speaker_party'].isin(['Vox'])), 'Speaker_wing'] = 'Más derecha'
    df.loc[(df['Speaker_party'].isin(['EAJ-PNV', 'JxCat-Junts', 'CDC', 'CiU'])), 'Speaker_wing'] = 'Regionalistas'
    df.loc[(df['Speaker_party'].isin(['Cs'])), 'Speaker_wing'] = 'Centro'
    party_groups = df.groupby('Speaker_wing')
    selected_speakers = df['Speaker_name'].value_counts()[:20].index.tolist()
    df = df[df['Speaker_name'].isin(selected_speakers)]
    df['Text'].apply(_clean_text_es)  # clean texts
    authors = df['Speaker_name'].unique()
    n_samples = []
    for author in authors:
        n_samples.append(df[df.Speaker_name == author].shape[0])
    df = df.sample(frac=1, random_state=42).groupby('Speaker_name').head(min(n_samples))
    df = df[df['Text'].notna()]
    texts = df["Text"].to_numpy()
    # creating stress encoding
    df['Stress'] = stress_encoding(texts)
    print('Creating LIWC encodings...')
    df['LIWC_gram'] = LIWC_encoding(texts, cat='gram', liwc_path=liwc_path)
    df['LIWC_obj'] = LIWC_encoding(texts, cat='obj', liwc_path=liwc_path)
    df['LIWC_cog'] = LIWC_encoding(texts, cat='cog', liwc_path=liwc_path)
    df['LIWC_feels'] = LIWC_encoding(texts, cat='feels', liwc_path=liwc_path)
    df['LIWC_CDI'] = LIWC_encoding(texts, cat='CDI', liwc_path=liwc_path)
    return df


# clean and lowercase the text
def _clean_text_es(text):
    # text = re.sub(r'\[\[(?:(?!\[|\])[\s\S])*\]\]', '', text)
    text = re.sub(r'ä', 'a', text)
    text = re.sub(r'ï', 'i', text)
    text = re.sub(r'ö', 'o', text)
    text = re.sub(r'¿', '¿ ', text)
    text = re.sub(r'¡', '¡ ', text)
    text = text.lower()
    return text


# print some info regarding the dataset
def dataset_info(df):
    print('Tot samples:', df.shape[0])
    print('Unique parties:', len(df['Speaker_party'].unique()), df['Speaker_party'].unique())
    print('Unique wings:', len(df['Speaker_wing'].unique()), df['Speaker_wing'].unique())
    authors = df['Speaker_name'].unique()
    n_samples, n_words = [], []
    for author in authors:
        n_samples.append(df[df.Speaker_name == author].shape[0])
        n_words.append(sum(df[df.Speaker_name == author]['Text'].str.split().str.len()))
    print('Author \t n_samples \t n_words')
    for i, author in enumerate(authors):
        print(author + '\t' + str(n_samples[i]) + '\t' + str(n_words[i]))
