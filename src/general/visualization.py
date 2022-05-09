from sklearn.decomposition import TruncatedSVD
from feature_extractor import featuresExtractor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from general.helpers import get_speaker_label
import textwrap
from explainability import ATI, CNI, ASI
import matplotlib.ticker as ticker
import collections
import numpy as np
from matplotlib.transforms import ScaledTranslation
from collections import Counter, OrderedDict
from itertools import cycle

colors_wing_es = {'Más izquierda': 'firebrick',
                  'Izquierda': 'salmon',
                  'Centro': 'lemonchiffon',
                  'Derecha': 'lightskyblue',
                  'Más derecha': 'royalblue',
                  'Regionalista': 'forestgreen'}

colors_wing_eng = {'Left': 'firebrick',
                   'Centre': 'gold',
                   'Right': 'cornflowerblue',
                   'Regionalist': 'forestgreen'}

colors_gender = {'M': 'goldenrod', 'F': 'mediumorchid'}

colors_party = {'PSOE': 'firebrick',
                'UP': 'firebrick',
                'Cs': 'gold',
                'PP': 'cornflowerblue',
                'PP-Foro': 'cornflowerblue',
                'EAJ-PNV': 'forestgreen',
                'JxCat-Junts': 'forestgreen'}


# change the bar dimension in barplot
# dimension = the dimension you want to change (height, weight)
# axis = the axis the barplot is oriented (x, y)
def _change_bar_dimension(ax, new_value, dimension, axis):
    for patch in ax.patches:
        if dimension == 'height':
            current_dimension = patch.get_height()
        else:
            current_dimension = patch.get_width()
        diff = current_dimension - new_value
        if dimension == 'height':
            patch.set_height(new_value)
        else:
            patch.set_width(new_value)
        if axis == 'y':
            patch.set_y(patch.get_y() + diff * .5)
        else:
            patch.set_x(patch.get_x() + diff * .5)


# change the wing labels from spanish to english
def _map_wing_es_en(labels_es):
    labels_eng = []
    for label in labels_es:
        if label == 'Izquierda':
            labels_eng.append('Left')
        elif label == 'Derecha':
            labels_eng.append('Right')
        elif label == 'Centro':
            labels_eng.append('Centre')
        else:
            labels_eng.append('Regionalist')
    return labels_eng


# plot of the number of words for each speaker
def plot_dataset_words(dataset):
    speakers = dataset['Speaker_name'].unique()
    n_words = []
    wings = _map_wing_es_en(get_speaker_label(dataset, 'wing'))
    for speaker in speakers:
        n_words.append(sum(dataset[dataset.Speaker_name == speaker]['Text'].str.split().str.len()))
    zipped_lists = zip(n_words, speakers, wings)
    sorted_pairs = sorted(zipped_lists, reverse=True)
    tuples = zip(*sorted_pairs)
    n_words, speakers, wings = [list(tuple) for tuple in tuples]
    speakers = [speaker.split(' ', 1)[0] for speaker in speakers]
    fig, ax = plt.subplots(1)
    sns.barplot(x=n_words, y=speakers, hue=wings, orient='h', palette=colors_wing_eng, dodge=False)
    ax.set_yticklabels([textwrap.fill(speaker, 12) for speaker in speakers])
    _change_bar_dimension(ax, .6, dimension='height', axis='y')
    plt.yticks(fontsize=9)
    plt.title('Total number of words for each speaker')
    plt.legend(loc="lower right")
    plt.show()


# plot the number of authors for each category (gender, birth, party)
def plot_dataset_categories(dataset):
    label_types = ['gender', 'birth', 'party']
    labels = []
    labels_count = []
    for label_type in label_types:
        authors_labels = get_speaker_label(dataset, label_type)
        count = OrderedDict(sorted(Counter(authors_labels).items()))
        labels.append(list(count.keys()))
        labels_count.append(list(count.values()))
    fig, axs = plt.subplots(1, len(labels))
    cycol = cycle(['turquoise', 'violet'])
    for plot_labels, plot_values, ax in zip(labels, labels_count, axs.flatten()):
        if 'party' in label_types and label_types.index('party') == labels.index(plot_labels):
            sns.barplot(x=plot_labels, y=plot_values, ax=ax, palette=colors_party, dodge=False)
        else:
            sns.barplot(x=plot_labels, y=plot_values, ax=ax, color=next(cycol), dodge=False)
        ax.set_yticks(np.arange(0, max(plot_values) + 1, step=1))
        ax.set_xticklabels([textwrap.fill(str(plot_label), 8) for plot_label in plot_labels],
                           rotation=75, size=7)
        _change_bar_dimension(ax, .6, dimension='width', axis='x')
    plt.show()


# plot each text sample in a 2D space using the given features, divided by wing
def plot_samples_wing(dataset):
    data = {'y': dataset['Speaker_name'].to_numpy(), 'texts': dataset['Text'].to_numpy(),
            'pos': dataset['POS-tags'].to_numpy(),
            'stress': dataset['Stress'].to_numpy(), 'liwc_gram': dataset['LIWC_gram'].to_numpy(),
            'liwc_obj': dataset['LIWC_obj'].to_numpy(), 'liwc_cog': dataset['LIWC_cog'].to_numpy(),
            'liwc_feels': dataset['LIWC_feels'].to_numpy()}
    print('Creating feature matrix...')
    feats = {'base_features': True, 'pos_tags': True, 'stress': True, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': True}
    X, _, _ = featuresExtractor(data, None, **feats)
    lsa = TruncatedSVD(n_components=2, random_state=42)  # cannot use PCA due to sparse matrix
    df_redux = pd.DataFrame(data=lsa.fit_transform(X), columns=['C1', 'C2'])
    wings = []
    for wing in dataset['Speaker_wing']:
        if wing == 'Izquierda':
            wings.append('Left')
        elif wing == 'Derecha':
            wings.append('Right')
        elif wing == 'Centro':
            wings.append('Centre')
        else:
            wings.append('Regionalist')
    df_redux['Wing'] = wings
    sns.scatterplot(data=df_redux, x="C1", y="C2", hue="Wing", palette=colors_wing_eng,
                    hue_order=colors_wing_eng.keys())
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Plot of transcripts by wing (with LSA)")
    plt.legend(loc='lower left')
    plt.show()


# plot the given index score for each speaker
def plot_index(dataset, index):
    assert index in ['ATI', 'CNI', 'ASI'], 'Plot available for indices: ATI, CNI, ASI.'
    speakers = dataset['Speaker_name'].unique()
    if index == 'ATI':
        results = ATI(dataset)
    elif index == 'CNI':
        results = CNI(dataset)
    else:
        results = ASI(dataset)
    wings = _map_wing_es_en(get_speaker_label(dataset, 'wing'))
    zipped_lists = zip(results, speakers, wings)
    sorted_pairs = sorted(zipped_lists, reverse=True)
    tuples = zip(*sorted_pairs)
    results, speakers, wings = [list(tuple) for tuple in tuples]
    speakers = [speaker.split(' ', 1)[0] for speaker in speakers]
    fig, ax = plt.subplots(1)
    sns.barplot(x=results, y=speakers, hue=wings, orient='h', palette=colors_wing_eng, dodge=False)
    ax.set_yticklabels([textwrap.fill(speaker, 12) for speaker in speakers])
    _change_bar_dimension(ax, .6, dimension='height', axis='y')
    plt.yticks(fontsize=8)
    # ax.set_xticklabels([10, 15, 20, 25, 30, 35, 40])
    if index == 'ATI':
        plt.title('Analytical Thinking Index for each speaker')
    elif index == 'CNI':
        plt.title('Categorical-Narrative Index for each speaker')
    else:
        plt.title('Adversarial Style Index for each speaker')
    plt.legend(loc="lower right")
    plt.show()


def plot_bestfit_spearman(dataset, index, res_methods):
    """
    Plot best-fit line for the given index and feat_res (for spearman experiment)
    :param dataset: the dataset
    :param index: the index to use (ATI, CNI, ASI)
    :param res_methods: list of tuple (output_file, method_name)
    """
    assert index in ['ATI', 'CNI', 'ASI'], 'Plot available for indices: ATI, CNI, ASI.'
    speakers = dataset['Speaker_name'].unique()
    speakers = [speaker.split(' ', 1)[0] for speaker in speakers]
    if index == 'ATI':
        index_results = ATI(dataset)
    elif index == 'CNI':
        index_results = CNI(dataset)
    else:
        index_results = ASI(dataset)
    fig, ax = plt.subplots()
    for res_method in res_methods:
        df_csv = pd.read_csv(res_method[0], sep=';', index_col=0).T
        df = pd.DataFrame({f'{index}': index_results})
        df[res_method[1]] = df_csv[res_method[1]].tolist()
        sns.regplot(x=f'{index}', y=f'{res_method[1]}', data=df, ax=ax, label=res_method[1])
        for i, author in enumerate(speakers):
            plt.text(df[index].iloc[i], df[res_method[1]].iloc[i], author, fontsize=8)
    # handles, labels = ax.get_legend_handles_labels()
    # n = len(feats_res)
    # plt.legend(handles=[(h1, h2) for h1, h2 in zip(handles[:n], handles[n:])], labels=labels[n:], fontsize=10, loc='upper left')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.ylabel(f"F1 result")
    plt.xlabel(index)
    plt.show()


def plot_anova(dataset, label_types, res_method):
    """
    Plot the F1 results of the given method, grouped by the given label type (for anova experiment).
    Create a single boxplot for each given label type.
    :param dataset: the dataset
    :param label_types: list of label types to group the F1 results into (wing, party, gender, birth)
    :param res_method: tuple (output_file, method_name)
    """
    assert all(label_type in ['wing', 'party', 'gender', 'birth'] for label_type in label_types), 'Label type must be: wing, party, gender, or birth.'
    res_csv = pd.read_csv(res_method[0], sep=';', index_col=0).T
    fig, axs = plt.subplots(1, len(label_types))
    for label_type, ax in zip(label_types, axs.flatten()):
        if label_type == 'wing':
            labels = _map_wing_es_en(get_speaker_label(dataset, label_type))
        else:
            labels = get_speaker_label(dataset, label_type)
        # consider only the groups with more than one element
        # clean both the labels and the results
        res_cleaned = res_csv.copy()
        idx_to_delete = [labels.index(label) for label in
                         [item for item, count in collections.Counter(labels).items() if count == 1]]
        labels_cleaned = [str(i) for j, i in enumerate(labels) if j not in idx_to_delete]
        for idx in idx_to_delete:
            res_cleaned = res_cleaned.drop([dataset['Speaker_name'].unique()[idx]])
        results = res_cleaned[res_method[1]]
        if label_type == 'gender':
            sns.boxplot(x=results, y=labels_cleaned, palette=colors_gender, ax=ax)
        elif label_type == 'wing':
            sns.boxplot(x=results, y=labels_cleaned, palette=colors_wing_eng, ax=ax)
        elif label_type == 'party':
            sns.boxplot(x=results, y=labels_cleaned, palette=colors_party, ax=ax)
        else:
            sns.boxplot(x=results, y=labels_cleaned, palette='deep', ax=ax)
        dx, dy = 0, 15
        offset = ScaledTranslation(dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans)
        ax.set_xlabel(f'{res_method[1]} (F1)')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
        for item in ax.yaxis.get_majorticklabels():
            item.set_transform(item.get_transform() + offset)
    plt.show()
