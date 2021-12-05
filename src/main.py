from general.utils import pickled_resource
from process_dataset import process_ParlaMint, dataset_info
from AA import build_AA_experiment
from general.visualization import plot_parties

if __name__ == '__main__':
    lang = 'es'  # options: es
    learner = 'svm'  # oprtions: svm lr random_forest

    print(f'--- Lang: {lang} ---')
    data_path = f"../pickles/dataset_ParlaMint_{lang}.pickle"
    dataset = pickled_resource(data_path, process_ParlaMint, lang=lang)
    #dataset_info(dataset)

    output_path = f'../output/{learner}_ParlaMint_{lang}.csv'  # csv file for the results
    pickle_path = f'../pickles/{learner}_preds_ParlaMint_{lang}.pickle'  # pickle file for the predictions

    feats = {'base_features': False,
             'pos_tags': True,
             'stress': True,
             'liwc_gram': True,
             'liwc_obj': True,
             'liwc_cog': True,
             'liwc_feels': True
             }

    build_AA_experiment(dataset, feats, learner, output_path, pickle_path)
    #plot_parties(dataset, feats)
