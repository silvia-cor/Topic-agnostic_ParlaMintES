from general.utils import pickled_resource
from process_dataset import process_ParlaMint
from SVM import build_SVM_experiment
from clustering import kmeans_clustering

if __name__ == '__main__':
    focus = 'party'  # options: name, party, gender
    lang = 'es'  # options: es

    print(f'--- Lang: {lang} | Focus: {focus} ---')
    data_path = f"../pickles/dataset_ParlaMint_{lang}_{focus}.pickle"
    dataset = pickled_resource(data_path, process_ParlaMint, focus=focus, lang=lang)
    print(f'Unique labels ({focus}): {len(dataset["unique_labels"])} {dataset["unique_labels"]}')
    print('#tot samples:', len(dataset['labels']))

    svm_output_path = f'../output/svm_ParlaMint_{lang}_{focus}.csv'  # csv file for the results
    svm_pickle_path = f'../pickles/svm_preds_ParlaMint_{lang}_{focus}.pickle'  # pickle file for the predictions

    feats = {'base_features': True,
             'pos_tags': True,
             'stress': True,
             'liwc_gram': True,
             'liwc_obj': True,
             'liwc_cog': True,
             'liwc_feels': True
             }



    #build_SVM_experiment(dataset, feats, svm_output_path, svm_pickle_path)
    kmeans_clustering(dataset, feats)
