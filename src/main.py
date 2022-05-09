from general.utils import pickled_resource
from process_dataset import process_ParlaMint_4wings, dataset_info
from ml_classification import build_multiclass_experiment, build_binary_experiment, party_status_experiment
from trans_classification import trans_classification, trans_party_status_experiment

if __name__ == '__main__':
    focus = 'name'  # options: name wing party gender birth party_status
    learner_name = 'svm'  # options: svm lr rf trans
    task = 'MC'  # options: BC MC

    print(f'--- Task: {task} ---')
    print(f'--- Focus: {focus} ---')
    data_path = f"../pickles/dataset_ParlaMint_es.pickle"
    liwc_path = f"../pickles/liwc_es.pickle"
    dataset = pickled_resource(data_path, process_ParlaMint_4wings, liwc_path=liwc_path)
    dataset_info(dataset)

    assert learner_name in ['svm', 'lr', 'rf', 'trans'], 'The learner must be svm, lr, rf, or trans'
    assert task in ['BC', 'MC'], 'The task must be either BinaryClassification or MulticlassClassification'
    assert focus in ['name', 'wing', 'party', 'gender', 'birth', 'party_status'], 'Focus must be name, wing, party, gender, birth, or party_status'

    if focus == 'party_status':
        output_path = f'../output/ParlaMint_AV_{learner_name}_name_partystatus.csv'
        pickle_path = f'../pickles/ParlaMint_AV_{learner_name}_name_partystatus.pickle'
    else:
        output_path = f'../output/ParlaMint_{task}_{learner_name}_{focus}.csv'  # csv file for the results
        pickle_path = f'../pickles/ParlaMint_{task}_{learner_name}_{focus}.pickle'  # pickle file for the predictions

    if learner_name in ['svm', 'lr', 'rf']:
        feats_series = [
            {'base_features': True, 'pos_tags': False, 'stress': False, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': False},
            {'base_features': True, 'pos_tags': True, 'stress': False, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': False},
            {'base_features': True, 'pos_tags': False, 'stress': True, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': False},
            {'base_features': True, 'pos_tags': False, 'stress': False, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': False},
            {'base_features': True, 'pos_tags': False, 'stress': False, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': False},
            {'base_features': True, 'pos_tags': False, 'stress': False, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': True},
            {'base_features': True, 'pos_tags': True, 'stress': True, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': True},
            {'base_features': False, 'pos_tags': True, 'stress': True, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': True},
            {'base_features': True, 'pos_tags': False, 'stress': True, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': True},
            {'base_features': True, 'pos_tags': True, 'stress': False, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': True},
            {'base_features': True, 'pos_tags': True, 'stress': True, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': True},
            {'base_features': True, 'pos_tags': True, 'stress': True, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': True},
            {'base_features': True, 'pos_tags': True, 'stress': True, 'liwc_gram': True, 'liwc_obj': False,
             'liwc_cog': True, 'liwc_feels': False}
        ]

        for feats in feats_series:
            if task == 'MC':
                build_multiclass_experiment(dataset, feats, learner_name, focus, output_path, pickle_path)
            else:
                if focus == 'party_status':
                    party_status_experiment(dataset, feats, learner_name, output_path, pickle_path)
                else:
                    build_binary_experiment(dataset, feats, learner_name, focus, output_path, pickle_path)
    else:
        if focus == 'party_status':
            trans_party_status_experiment(dataset, output_path, pickle_path)
        else:
            trans_classification(dataset, focus, output_path, pickle_path, task)
