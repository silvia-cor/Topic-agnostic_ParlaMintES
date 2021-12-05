import pandas as pd
import researchpy as rp
import statsmodels as stats
import scipy
import numpy as np
from feature_extractor import featuresExtractor
from sklearn.mixture import GaussianMixture
from process_dataset import process_ParlaMint
from general.utils import pickled_resource


def mix_model(dataset, feats, focus='party'):
    assert focus in ['party'], 'Only implementation with focus = party'
    grouped_dataset = dataset.groupby('Speaker_party')
    for party, mini_dataset in grouped_dataset:
        data = {'y': mini_dataset['Speaker_name'], 'texts': mini_dataset['Text'].to_numpy(),
                'pos': mini_dataset['POS-tags'].to_numpy(),
                'stress': mini_dataset['Stress'].to_numpy(), 'liwc_gram': mini_dataset['LIWC_gram'].to_numpy(),
                'liwc_obj': mini_dataset['LIWC_obj'].to_numpy(), 'liwc_cog': mini_dataset['LIWC_cog'].to_numpy(),
                'liwc_feels': mini_dataset['LIWC_feels'].to_numpy()}
        X, _ = featuresExtractor(data, None, **feats)
        gm = GaussianMixture(n_components=1, random_state=42).fit(X.toarray())
        gm_mean = gm.means_[0]
        gm_stdv = np.sqrt(np.trace(gm.covariances_[0]))
        print(gm_mean)
        print(gm_stdv)


lang = 'es'
data_path = f"../pickles/dataset_ParlaMint_{lang}.pickle"
dataset = pickled_resource(data_path, process_ParlaMint, lang=lang)
feats = {'base_features': True,
             'pos_tags': True,
             'stress': True,
             'liwc_gram': True,
             'liwc_obj': True,
             'liwc_cog': True,
             'liwc_feels': True
             }
mix_model(dataset, feats)

