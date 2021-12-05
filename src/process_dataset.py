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


# TODO: add Italian option
# create the dataframe
def process_ParlaMint(data_path='../dataset/ParlaMint-ES.conllu', file_path='/ParlaMint-ES.csv', lang='es'):
    print('--- CREATING DATASET PICKLE ---')
    assert lang in ['es', 'it'], 'The project language must be either es (spanish) or it (italian).'
    if not os.path.isfile(data_path + file_path):
        create_ParlaMint_csv(data_path)
    else:
        print('PARLAMINT csv file found')
    df = pd.read_csv(data_path + file_path, sep='\t')  # read csv file
    if lang == 'es':
        df = df[df.Speaker_role != 'Chairperson']  # delete ChairPerson entries
        df = df[df.groupby('Speaker_party').Speaker_name.transform('count') >= 300]
        party_groups = df.groupby('Speaker_party')
        selected_speakers = []
        for name, group in party_groups:
            selected_speakers.extend(group['Speaker_name'].value_counts()[:2].index.tolist())
        df = df[df['Speaker_name'].isin(selected_speakers)]
        # resolve ambiguous Speaker_party entries
        #.loc[(df['Speaker_name'] == 'Martínez Seijo, María Luz') & (df['Speaker_party'] == 'PP;PSOE;UP'), 'Speaker_party'] = 'PSOE'
        #df.loc[(df['Speaker_party'] == 'PP;PSOE') & (df['Speaker_name'].isin(['González Veracruz, María',
                                                                              #'Martínez Seijo, María Luz',
                                                                              #'Martín González, María Guadalupe'])), 'Speaker_party'] = 'PSOE'
        #df = df.drop(df[(df['Speaker_party'] == 'PP;PSOE') & (df['Speaker_name'] == 'Rodríguez Ramos, María Soraya')].index)
        # assign Speaker_wing
        df["Speaker_wing"] = np.nan
        df.loc[(df['Speaker_party'].isin(['PSOE', 'PSC-PSOE'])), 'Speaker_wing'] = 'Izquierda'
        df.loc[(df['Speaker_party'].isin(['PP', 'PP-Foro'])), 'Speaker_wing'] = 'Derecha'
        df.loc[(df['Speaker_party'].isin(['ERC-S', 'EH Bildu', 'ERC-CATSÍ', 'UP'])), 'Speaker_wing'] = 'Más izquierda'
        df.loc[(df['Speaker_party'].isin(['Vox'])), 'Speaker_wing'] = 'Más derecha'
        df.loc[(df['Speaker_party'].isin(['EAJ-PNV', 'JxCat-Junts', 'CDC', 'CiU'])), 'Speaker_wing'] = 'Regionalistas'
        df.loc[(df['Speaker_party'].isin(['Cs'])), 'Speaker_wing'] = 'Centro'
        df['Text'] = df['Text'].apply(_clean_text_es)  # clean texts
    df = df[df['Text'].notna()]
    texts = df["Text"].to_numpy()
    # creating stress encoding
    df['Stress'] = stress_encoding(texts)
    print('Creating LIWC encodings...')
    df['LIWC_gram'] = LIWC_encoding(texts, cat='gram')
    df['LIWC_obj'] = LIWC_encoding(texts, cat='obj')
    df['LIWC_cog'] = LIWC_encoding(texts, cat='cog')
    df['LIWC_feels'] = LIWC_encoding(texts, cat='feels')
    return df


def _clean_text_es(text):
    # text = re.sub(r'\[\[(?:(?!\[|\])[\s\S])*\]\]', '', text)
    text = re.sub(r'ä', 'a', text)
    text = re.sub(r'ï', 'i', text)
    text = re.sub(r'ö', 'o', text)
    text = re.sub(r'¿', '¿ ', text)
    text = re.sub(r'¡', '¡ ', text)
    text = text.lower()
    return text


def dataset_info(df):
    groups = df.groupby('Speaker_name')
    len_groups = [(name, group['Text'].str.len().sum()) for name, group in groups]
    print('Tot samples:', df.shape[0])
    print('Unique speakers:', len(df['Speaker_name'].unique()), df['Speaker_name'].unique())
    print('Unique parties:', len(df['Speaker_party'].unique()), df['Speaker_party'].unique())
    print('Unique wings:', len(df['Speaker_wing'].unique()), df['Speaker_wing'].unique())
    print('Min speaker:', min(len_groups, key=lambda x: x[1]))
    print('Max speaker:', max(len_groups, key=lambda x: x[1]))
    print(f'Mean among speakers: {np.mean([x[1] for x in len_groups]):.2f}')
