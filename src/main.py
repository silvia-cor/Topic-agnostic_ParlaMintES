from general.utils import pickled_resource
from process_dataset import process_ParlaMint
import random
from SVM import build_SVM_experiment

if __name__ == '__main__':
    focus = 'name'  # options: name, party, gender
    lang = 'es'  # options: es, it

    print(f'--- Lang: {lang} | Focus: {focus} ---')
    data_path = f"../pickles/dataset_ParlaMint_{lang}_{focus}.pickle"
    dataset = pickled_resource(data_path, process_ParlaMint, focus=focus, lang=lang)
    print(f'Unique labels ({focus}):', len(dataset['unique_labels']))
    print('#tot samples:', len(dataset['labels']))

    # n = random.randint(0, 5789)
    # print(dataset['texts'][n])
    # author = dataset['labels'][n]
    # print(dataset['unique_labels'][author])
    # print(dataset['pos_tags_texts'][n])
    # print(dataset['stress_texts'][n])
    # print(dataset['liwc_gram_texts'][n])
    # print(dataset['liwc_obj_texts'][n])
    # print(dataset['liwc_cog_texts'][n])
    # print(dataset['liwc_feels_texts'][n])

    svm_path = f'../output/exp_svm_ParlaMint_{lang}_{focus}.csv'

    svm_feats = {'base_features': True,
                 'pos_tags': False,
                 'stress': False,
                 'liwc_gram': False,
                 'liwc_obj': False,
                 'liwc_cog': False,
                 'liwc_feels': False
                 }

    build_SVM_experiment(dataset, svm_feats, svm_path)
