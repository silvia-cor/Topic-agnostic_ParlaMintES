import os
import pickle
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


# pickle the output of a function
def pickled_resource(pickle_path: str, generation_func: callable, *args, **kwargs):
    if pickle_path is None:
        return generation_func(*args, **kwargs)
    else:
        if os.path.exists(pickle_path):
            return pickle.load(open(pickle_path, 'rb'))
        else:
            instance = generation_func(*args, **kwargs)
            os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
            pickle.dump(instance, open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            return instance


# tokenize text without punctuation
def tokenize_nopunct(text):
    unmod_tokens = nltk.word_tokenize(text)
    return [token.lower() for token in unmod_tokens if
            any(char.isalpha() for char in token)]  # checks whether all the chars are alphabetic


# return list of function words
def get_function_words(lang):
    languages = {'es': 'spanish',
                 'it': 'italian'
                 }
    if lang not in languages:
        raise ValueError('{} not in scope!'.format(lang))
    else:
        return stopwords.words(languages[lang])


# find the Longest Common Prefix among two words (value)
def find_LCP(x, y):
    prefix_length = 0
    for i in range(0, min(len(x), len(y))):
        if x[i] == y[i]:
            prefix_length += 1
        else:
            break
    return prefix_length


def divide_words_asterisk(words):
    list_w, list_w_ast = [], []
    for w in words:
        if len(w) == 0:
            print(w)
        if w[-1] == "*":
            list_w_ast.append(w)
        else:
            list_w.append(w)
    return list_w, list_w_ast

