import pandas as pd
from general.helpers import adjust_dataset, get_speaker_label
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import scipy.stats as stats
import numpy as np
import collections


def ANOVA_test(dataset, label_type, res_files):
    """
    Perform the ANOVA test on the F1 results in the AV task.
    :param dataset: the dataset
    :param label_type: the label by which grouping of the results
    :param res_files: files path(s) where to take the F1 results
    """
    assert label_type in ['wing', 'party', 'gender', 'birth'], 'The grouping for ANOVA test can be done by wing, party gender or birth'
    labels_tot = get_speaker_label(dataset, label_type)
    idx_to_delete = [labels_tot.index(label) for label in
                     [item for item, count in collections.Counter(labels_tot).items() if count == 1]]
    labels = [i for j, i in enumerate(labels_tot) if j not in idx_to_delete]
    for res_file in res_files:
        res_csv = pd.read_csv(res_file, sep=';', index_col=0).T
        for idx in idx_to_delete:
            res_csv = res_csv.drop([dataset['Speaker_name'].unique()[idx]])
        res_csv['label'] = labels
        # renaming columns, or the '+' and '-' might trigger expressions...
        res_csv.columns = res_csv.columns.str.replace("[ + ]", "", regex=True)
        res_csv.columns = res_csv.columns.str.replace("[ - ]", "_", regex=True)
        for feature_type in res_csv.columns:
            if feature_type != 'label':
                print(f'ANOVA test among {label_type} and {feature_type}')
                model = ols(f'{feature_type} ~  C(label)', data=res_csv).fit()
                print('Check for normality:', stats.shapiro(model.resid)[1])  # must be non-significant
                groups = res_csv.groupby('label')[feature_type].apply(list)
                print('Check for homogeneity of variance:', stats.levene(*groups)[1])  # must be non-significant
                cat = res_csv.groupby('label')[feature_type].apply(list)
                anova_res = f_oneway(*cat)
                print(f'P-Value Anova: {anova_res[1]:.3f}')
                # if p-value < 0.05, perform Tukey HSD test (in order to see the group(s) causing the difference)
                if anova_res[1] < 0.05:
                    comp = mc.MultiComparison(res_csv[feature_type], res_csv['label'])
                    post_hoc_res = comp.tukeyhsd()
                    print(post_hoc_res.summary())
                print('\n')


def spearman_correlation(dataset, index_name, res_files):
    """
    Compute the Spearman correlation among the index of choice and the F1 results in the AV task
    :param dataset: the dataset
    :param index_name: index to use for correlation (ATI, CNI, ASI, P-NI, P+NI)
    :param res_files: files path(s) where to take the F1 results
    """
    assert index_name in ['ATI', 'CNI', 'ASI', 'P-NI', 'P+NI'], 'Index must be ATI, CNI, ASI, P-NI or P+NI'
    if index_name == 'ATI':
        labels = ATI(dataset)
    elif index_name == 'ASI':
        labels = ASI(dataset)
    elif index_name == 'P-NI':
        labels = Pos_Neg_Index(dataset, vs=True)
    elif index_name == 'P+NI':
        labels = Pos_Neg_Index(dataset, vs=False)
    else:
        labels = CNI(dataset)
    for res_file in res_files:
        res_csv = pd.read_csv(res_file, sep=';', index_col=0).T
        for feature_type in res_csv.columns:
            if feature_type != 'label':
                spearman = stats.spearmanr(labels, res_csv[feature_type])
                print(f'Correlation among {index_name} and {feature_type}: r = {spearman[0]:.3f}  p-value= {spearman[1]:.3f}')


# compute the Analytical Thinking Index
def ATI(dataset):
    authors = dataset['Speaker_name'].unique()
    data = adjust_dataset(dataset, focus='name')
    CDI_results = []
    for author in authors:
        # print('CDI for', author)
        texts = ' '.join([data['texts'][i] for i, label in enumerate(data['y']) if label == author])
        liwc_texts = ' '.join([data['liwc_CDI'][i] for i, label in enumerate(data['y']) if label == author])
        pos_texts = ' '.join([data['pos'][i] for i, label in enumerate(data['y']) if label == author])
        tot_words = len(texts.split())
        # print('Tot.words:', tot_words)
        articles = liwc_texts.count('Articulo') * 100 / tot_words
        preps = liwc_texts.count('Prepos') * 100 / tot_words
        pronper = liwc_texts.count('PronPer') * 100 / tot_words
        pronimp = liwc_texts.count('PronImp') * 100 / tot_words
        aux = liwc_texts.count('VerbAux') * 100 / tot_words
        conj = liwc_texts.count('Conjunc') * 100 / tot_words
        adv = liwc_texts.count('Adverb') * 100 / tot_words
        neg = liwc_texts.count('Negacio') * 100 / tot_words
        CDI = articles + preps - pronper - pronimp - aux - conj - adv - neg
        # print(f'CDI= {CDI:.2f}\n')
        CDI_results.append(CDI)
    return CDI_results


# compute Categorical-vs-Narrative Index
def CNI(dataset):
    authors = dataset['Speaker_name'].unique()
    data = adjust_dataset(dataset, focus='name')
    CNI_results = []
    for author in authors:
        # print('CNI for', author)
        liwc_texts = ' '.join([data['liwc_CDI'][i] for i, label in enumerate(data['y']) if label == author])
        pos_texts = ' '.join([data['pos'][i] for i, label in enumerate(data['y']) if label == author])
        texts = ' '.join([data['texts'][i] for i, label in enumerate(data['y']) if label == author])
        tot_words = len(texts.split())
        # print('Tot.words:', tot_words)
        nouns = pos_texts.count('NOUN') * 100 / tot_words
        adjs = pos_texts.count('ADJ') * 100 / tot_words
        preps = liwc_texts.count('Prepos') * 100 / tot_words
        verbs = pos_texts.count('VERB') * 100 / tot_words
        adverbs = liwc_texts.count('Adverb') * 100 / tot_words
        pronpers = liwc_texts.count('PronPer') * 100 / tot_words
        CNI = nouns + adjs + preps - verbs - adverbs - pronpers
        # print(f'CNI= {CNI:.2f}\n')
        CNI_results.append(CNI)
    CNI_single_texts = []
    for i, liwc_text in enumerate(data['liwc_CDI']):
        tot_words = len(data['texts'][i].split())
        pos_text = data['pos'][i]
        nouns = pos_text.count('NOUN') * 100 / tot_words
        adjs = pos_text.count('ADJ') * 100 / tot_words
        preps = liwc_text.count('Prepos') * 100 / tot_words
        verbs = pos_text.count('VERB') * 100 / tot_words
        adverbs = liwc_text.count('Adverb') * 100 / tot_words
        pronpers = liwc_text.count('PronPer') * 100 / tot_words
        CNI = nouns + adjs + preps - verbs - adverbs - pronpers
        CNI_single_texts.append(CNI)
    max_CNI_idx = np.argpartition(np.array(CNI_single_texts), -10)[-10:]
    min_CNI_idx = np.array(CNI_single_texts).argsort()[:10]
    # print('Texts with max CNI')
    # for idx in max_CNI_idx:
    #     print(data['texts'][idx])
    # print('Texts with min CNI')
    # for idx in min_CNI_idx:
    #     print(data['texts'][idx])
    return CNI_results


# compute Adversarial Style Index
def ASI(dataset):
    authors = dataset['Speaker_name'].unique()
    data = adjust_dataset(dataset, focus='name')
    index_results = []
    for author in authors:
        liwc_texts = ' '.join([data['liwc_gram'][i] for i, label in enumerate(data['y']) if label == author])
        texts = ' '.join([data['texts'][i] for i, label in enumerate(data['y']) if label == author])
        tot_words = len(texts.split())
        yo = (liwc_texts.count('Yo') * 100 / tot_words)
        nos = (liwc_texts.count('Nosotro') * 100 / tot_words)
        tu = (liwc_texts.count('TuUtd') * 100 / tot_words)
        uds = (liwc_texts.count('VosUtds') * 100 / tot_words)
        index = (uds + tu) / (nos + yo + uds + tu)
        # print(f'ASI for {author}: {index:.2f}')
        index_results.append(index)
    return index_results


# compute Pos-Neg (vs=True) or Pos+Neg (vs=False) Index
def Pos_Neg_Index(dataset, vs):
    authors = dataset['Speaker_name'].unique()
    data = adjust_dataset(dataset, focus='name')
    index_results = []
    for author in authors:
        liwc_texts = ' '.join([data['liwc_feels'][i] for i, label in enumerate(data['y']) if label == author])
        texts = ' '.join([data['texts'][i] for i, label in enumerate(data['y']) if label == author])
        tot_words = len(texts.split())
        if vs:
            index = (liwc_texts.count('EmoPos') * 100 / tot_words) - (liwc_texts.count('EmoNeg') * 100 / tot_words)
            print(f'Pos-Neg Index for {author}: {index:2f}')
        else:
            index = (liwc_texts.count('EmoPos') * 100 / tot_words) + (liwc_texts.count('EmoNeg') * 100 / tot_words)
            print(f'Pos+Neg Index for {author}: {index:2f}')
        index_results.append(index)
    return index_results
