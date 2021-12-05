from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from feature_extractor import featuresExtractor
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import scipy.stats as stats

colors_party = {'Más izquierda': 'firebrick',
                   'Izquierda': 'salmon',
                   'Centro': 'lemonchiffon',
                   'Derecha': 'lightskyblue',
                   'Más derecha': 'royalblue',
                   'Regionalistas': 'forestgreen'}


def plot_dataset(database):
    speakers = database['Speaker_name'].to_list()
    speakers = [speaker.split(' ', 1)[0] for speaker in speakers]
    database['Speaker_name'] = speakers
    sns.countplot(y="Speaker_name", hue='Speaker_wing', data=database, palette=colors_party, dodge=False)
    plt.yticks(fontsize=5.5)
    plt.title('Count of samples for each speaker')
    plt.legend(loc="lower right")
    plt.show()


def plot_parties(database, feats):
    data = {'y': database['Speaker_name'].to_numpy(), 'texts': database['Text'].to_numpy(), 'pos': database['POS-tags'].to_numpy(),
            'stress': database['Stress'].to_numpy(), 'liwc_gram': database['LIWC_gram'].to_numpy(),
               'liwc_obj': database['LIWC_obj'].to_numpy(), 'liwc_cog': database['LIWC_cog'].to_numpy(), 'liwc_feels': database['LIWC_feels'].to_numpy()}
    print('Creating feature matrix...')
    X, _ = featuresExtractor(data, None, **feats)
    lsa = TruncatedSVD(n_components=2, random_state=42)  # cannot use PCA due to sparse matrix
    df_redux = pd.DataFrame(data=lsa.fit_transform(X), columns=['C1', 'C2'])
    df_redux['Party'] = database['Speaker_wing']
    sns.scatterplot(data=df_redux, x="C1", y="C2", hue="Party", palette=colors_party, hue_order=colors_party.keys())
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Plot of transcripts by parties (with LSA)")
    plt.legend(loc='lower left')
    plt.show()

