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


def kmeans_clustering(dataset, feats):
    data = {'unique_labels': dataset['unique_labels'], 'y': dataset['labels'], 'texts': dataset['texts'],
            'pos': dataset['pos_tags_texts'],
            'stress': dataset['stress_texts'], 'liwc_gram': dataset['liwc_gram_texts'],
            'liwc_obj': dataset['liwc_obj_texts'], 'liwc_cog': dataset['liwc_cog_texts'],
            'liwc_feels': dataset['liwc_feels_texts']}
    print('Creating feature matrix...')
    X, _ = featuresExtractor(data, None, **feats)
    #best_k, best_model = _get_best_k_model(X.toarray(), max_k=10)
    best_k, best_model = _get_best_k_model(X, max_k=10)
    print('Best k:', best_k)
    #cluster_labels = best_model.predict(X.toarray())
    cluster_labels = best_model.predict(X)
    _visualize_clustering(X, [data['unique_labels'][label] for label in data['y']], cluster_labels)
    df_pres = pd.DataFrame(data=[(0 for i in range(len(data['unique_labels']))) for i in range(best_k)], columns=data['unique_labels'])
    for party in range(len(data['unique_labels'])):
        idxs = np.where(data['y'] == party)[0]
        for k in list(range(best_k)):
            n = 0
            for idx in idxs:
                if cluster_labels[idx] == k:
                    n += 1
            df_pres.iloc[k, party] = n/len(idxs)
    #tot = list(df_pres.sum(axis=1))
    #df_pres['Tot.'] = [item/len(data['y']) for item in tot]
    print(df_pres)
    df_chi2 = pd.DataFrame(data=[(0 for i in range(len(df_pres.columns))) for i in range(len(df_pres.columns))], columns=list(df_pres.columns))
    df_chi2.set_index(pd.Index(list(df_pres.columns)), inplace=True)
    for n, i in enumerate(list(df_pres.columns)):
        for m, j in enumerate(list(df_pres.columns)):
            if i != j:
                #stat, p_val = stats.wilcoxon(np.array(df_pres[i]), np.array(df_pres[j]))
                #df_chi2.loc[i, j] = p_val
                diff = np.diff(np.array([df_pres[i], df_pres[j]]), axis=0)
                print(diff)
                diff = [item for sublist in diff for item in sublist]
                print(diff)
                diff = np.abs(diff)
                print(diff)
                ci = stats.norm.interval(alpha=0.95, loc=np.mean(diff), scale=stats.sem(diff))
                print(ci)
                if ci[0] < 0 < ci[1]:
                    print(f"{data['unique_labels'][n]} and {data['unique_labels'][m]} are not different")
                else:
                    print(f"{data['unique_labels'][n]} and {data['unique_labels'][m]} are different")
    #print(df_chi2)
    #_visualize_chi2(df_chi2)


def _visualize_clustering(X, labels, cluster_labels):
    colors_party = {'Más izquierda': 'firebrick',
                   'Izquierda': 'salmon',
                   'Centro': 'lemonchiffon',
                   'Derecha': 'lightskyblue',
                   'Más derecha': 'royalblue',
                   'Regionalistas': 'forestgreen'}
    lsa = TruncatedSVD(n_components=2, random_state=42)  # cannot use PCA due to sparse matrix
    # tsne = TSNE(n_components=2, random_state=42)
    df_redux = pd.DataFrame(data=lsa.fit_transform(X), columns=['C1', 'C2'])
    df_redux['Party'] = labels
    df_redux['Cluster'] = cluster_labels
    sns.scatterplot(data=df_redux, x="C1", y="C2", hue="Party", palette=colors_party, hue_order=colors_party.keys())
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Plot of transcripts by parties (with LSA)")
    plt.legend(loc='lower left')
    plt.show()
    sns.scatterplot(data=df_redux, x="C1", y="C2", hue="Cluster")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("KMeans clustering (with LSA)")
    plt.legend(loc='lower left')
    plt.show()


def _get_best_k_model(X, max_k):
    models = []
    inertias = []
    ks = list(range(2, max_k+1))
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42).fit(X)
        models.append(model)
        inertias.append(model.inertia_)
    kn = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
    best_k = kn.knee
    return best_k, models[ks.index(best_k)]
    # ks = list(range(2, max_k+1))
    # models = [GaussianMixture(k, random_state=42).fit(X) for k in ks]
    # bics = [model.aic(X) for model in models]
    # print(bics)
    # best_idx = bics.index(min(bics))
    # print(best_idx)
    # best_k = ks[best_idx]
    # print(best_k)
    # best_model = models[best_idx]
    #return best_k, best_model



def _visualize_chi2(chi2_result):
    sns.set(font_scale=0.7)
    sns.heatmap(chi2_result, annot=True)
    plt.title('Wilcoxon test result')
    plt.show()

#def _visualize_ANOVA(x, y, data):

