from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import nltk
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize
from collections import Counter
from general.utils import tokenize_nopunct, get_function_words


# ------------------------------------------------------------------------
# feature extraction methods
# ------------------------------------------------------------------------


# extract the frequency (L1x1000) of each function word used in the documents
def _function_words_freq(documents, function_words):
    features = []
    for text in documents:
        mod_tokens = tokenize_nopunct(text)
        freqs = nltk.FreqDist(mod_tokens)
        nwords = len(mod_tokens)
        funct_words_freq = [freqs[function_word] / nwords for function_word in function_words]
        features.append(funct_words_freq)
    f = csr_matrix(features)
    return f


# extract the frequencies (L1x1000) of the words' lengths used in the documents,
# following the idea behind Mendenhall's Characteristic Curve of Composition
def _word_lengths_freq(documents, upto):
    features = []
    for text in documents:
        mod_tokens = tokenize_nopunct(text)
        nwords = len(mod_tokens)
        tokens_len = [len(token) for token in mod_tokens]
        tokens_count = []
        for i in range(1, upto + 1):
            tokens_count.append((sum(j >= i for j in tokens_len)) / nwords)
        features.append(tokens_count)
    f = csr_matrix(features)
    return f


# extract lengths of the sentences, ie. number of words in the sentence
def _sentence_lengths_freq(documents, upto):
    features = []
    for text in documents:
        sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
        nsent = len(sentences)
        sent_len = []
        sent_count = []
        for sentence in sentences:
            mod_tokens = tokenize_nopunct(sentence)
            sent_len.append(len(mod_tokens))
        for i in range(1, upto + 1):
            sent_count.append((sum(j >= i for j in sent_len)) / nsent)
        features.append(sent_count)
    f = csr_matrix(features)
    return f


# vectorize the documents with tfidf and select the best features
def _vector_select(tr, te, y_tr, analyzer_type, n_min, n_max, fs):
    vectorizer = TfidfVectorizer(analyzer=analyzer_type, ngram_range=(n_min, n_max), sublinear_tf=True)
    f_train = vectorizer.fit_transform(tr)
    ft_names = vectorizer.get_feature_names_out()
    if fs:
        num_feats = int(f_train.shape[1] * fs)
        selector = SelectKBest(chi2, k=num_feats)
        f_train = selector.fit_transform(f_train, y_tr)
        idxs = selector.get_support(indices=True)
        ft_names = [ft_names[idx] for idx in idxs]
    if te is None:
        return f_train, ft_names
    else:
        f_test = vectorizer.transform(te)
        if fs:
            f_test = selector.transform(f_test)
        return f_train, f_test, ft_names


# ------------------------------------------------------------------------
# Feature Extractor
# ------------------------------------------------------------------------
def featuresExtractor(tr_data, te_data,
                      base_features=True,
                      pos_tags=False,
                      stress=False,
                      liwc_gram=False,
                      liwc_obj=False,
                      liwc_cog=False,
                      liwc_feels=False
                      ):
    """
    For each feature type, the corresponding function is called and a csr_matrix is created.
    The matrix is then added orizontally (hstack) to the final matrix.
    Train and test are kept separate to properly fit on training set for n-grams vectorization and feature selection.
    :param tr_data: train set
    :param te_data: test set
    :param base_features: BaseFeature set (function words, word lengths, sentence lengths)
    :param pos_tags: POS set (encoding of PoS-tags and extraction of word n-grams TfIdf)
    :param stress: STRESS set (encoding of stressed/unstressed syllables and extraction of char n-grams TfIdf)
    :param liwc_gram: LIWC_GRAM set (encoding of LIWC GRAM categories and extraction of word n-grams TfIdf)
    :param liwc_obj: LIWC_OBJ set (encoding of LIWC OBJ categories and extraction of word n-grams TfIdf) (not used)
    :param liwc_cog: LIWC_COG set (encoding of LIWC COG categories and extraction of word n-grams TfIdf)
    :param liwc_feels: LIWC_FEELS set (encoding of FEELS GRAM categories and extraction of word n-grams TfIdf)
    :return:
    """

    # final matrixes of features
    # initialize the right number of rows, or hstack won't work
    X_tr = csr_matrix((len(tr_data['texts']), 0))
    if te_data:
        X_te = csr_matrix((len(te_data['texts']), 0))
    else:
        X_te = None

    fw = get_function_words(lang='es')
    features_names = []

    if base_features:
        # function words
        f = normalize(_function_words_freq(tr_data['texts'], fw))
        X_tr = hstack((X_tr, f))
        if te_data:
            f = normalize(_function_words_freq(te_data['texts'], fw))
            X_te = hstack((X_te, f))
        features_names.extend(fw)
        print(f'task function words (#features={f.shape[1]}) [Done]')

        # word lenghts
        lens = [len(word) for word in tokenize_nopunct(''.join(tr_data['texts']))]
        upto = max([k for k, v in Counter(lens).items() if
                    v >= 5])  # find longest word in training set appearing at least 10 times
        f = normalize(_word_lengths_freq(tr_data['texts'], upto))
        X_tr = hstack((X_tr, f))
        if te_data:
            f = normalize(_word_lengths_freq(te_data['texts'], upto))
            X_te = hstack((X_te, f))
        for length in range(1, upto):
            features_names.append(f'w_len_{length}')
        print(f'task word lengths (#features={f.shape[1]}) [Done]')

        # sentence lengths
        sents = [t.strip() for t in nltk.tokenize.sent_tokenize(' '.join(tr_data['texts'])) if t.strip()]
        lens = [len(tokenize_nopunct(t)) for t in sents]
        upto = max([k for k, v in Counter(lens).items() if
                    v >= 5])  # find longest sent in training set appearing at least 10 times
        f = normalize(_sentence_lengths_freq(tr_data['texts'], upto))
        X_tr = hstack((X_tr, f))
        if te_data:
            f = normalize(_sentence_lengths_freq(te_data['texts'], upto))
            X_te = hstack((X_te, f))
        for length in range(1, upto):
            features_names.append(f's_len_{length}')
        print(f'task sentence lengths (#features={f.shape[1]}) [Done]')

    if pos_tags:
        if te_data:
            f_tr, f_te, ft_names = _vector_select(tr_data['pos'], te_data['pos'], tr_data['y'], analyzer_type='word',
                                                  n_min=1, n_max=3, fs=None)
            X_te = hstack((X_te, f_te))
        else:
            f_tr, ft_names = _vector_select(tr_data['pos'], None, tr_data['y'], analyzer_type='word', n_min=1, n_max=3,
                                            fs=None)
        X_tr = hstack((X_tr, f_tr))
        features_names.extend(ft_names)
        print(f'task pos-tags encoding (#features={f_tr.shape[1]}) [Done]')

    if stress:
        if te_data:
            f_tr, f_te, ft_names = _vector_select(tr_data['stress'], te_data['stress'], tr_data['y'],
                                                  analyzer_type='char',
                                                  n_min=1, n_max=7, fs=None)
            X_te = hstack((X_te, f_te))
        else:
            f_tr, ft_names = _vector_select(tr_data['stress'], None, tr_data['y'], analyzer_type='char', n_min=1,
                                            n_max=7, fs=None)
        X_tr = hstack((X_tr, f_tr))
        features_names.extend(ft_names)
        print(f'task stress encoding (#features={f_tr.shape[1]}) [Done]')

    if liwc_gram:
        if te_data:
            f_tr, f_te, ft_names = _vector_select(tr_data['liwc_gram'], te_data['liwc_gram'], tr_data['y'],
                                                  analyzer_type='word',
                                                  n_min=1, n_max=3, fs=0.1)
            X_te = hstack((X_te, f_te))
        else:
            f_tr, ft_names = _vector_select(tr_data['liwc_gram'], None, tr_data['y'], analyzer_type='word', n_min=1,
                                            n_max=3, fs=0.1)
        X_tr = hstack((X_tr, f_tr))
        features_names.extend(ft_names)
        print(f'task liwc_gram encoding (#features={f_tr.shape[1]}) [Done]')

    if liwc_obj:
        if te_data:
            f_tr, f_te, ft_names = _vector_select(tr_data['liwc_obj'], te_data['liwc_obj'], tr_data['y'],
                                                  analyzer_type='word',
                                                  n_min=1, n_max=3, fs=0.1)
            X_te = hstack((X_te, f_te))
        else:
            f_tr, ft_names = _vector_select(tr_data['liwc_obj'], None, tr_data['y'], analyzer_type='word', n_min=1,
                                            n_max=3, fs=0.1)
        X_tr = hstack((X_tr, f_tr))
        features_names.extend(ft_names)
        print(f'task liwc_cog encoding (#features={f_tr.shape[1]}) [Done]')

    if liwc_cog:
        if te_data:
            f_tr, f_te, ft_names = _vector_select(tr_data['liwc_cog'], te_data['liwc_cog'], tr_data['y'],
                                                  analyzer_type='word',
                                                  n_min=1, n_max=3, fs=0.1)
            X_te = hstack((X_te, f_te))
        else:
            f_tr, ft_names = _vector_select(tr_data['liwc_cog'], None, tr_data['y'], analyzer_type='word', n_min=1,
                                            n_max=3, fs=0.1)
        X_tr = hstack((X_tr, f_tr))
        features_names.extend(ft_names)
        print(f'task liwc_cog encoding (#features={f_tr.shape[1]}) [Done]')

    if liwc_feels:
        if te_data:
            f_tr, f_te, ft_names = _vector_select(tr_data['liwc_feels'], te_data['liwc_feels'], tr_data['y'],
                                                  analyzer_type='word', n_min=1, n_max=3, fs=0.1)
            X_te = hstack((X_te, f_te))
        else:
            f_tr, ft_names = _vector_select(tr_data['liwc_feels'], None, tr_data['y'], analyzer_type='word', n_min=1,
                                            n_max=3, fs=0.1)
        X_tr = hstack((X_tr, f_tr))
        features_names.extend(ft_names)
        print(f'task liwc_feels encoding (#features={f_tr.shape[1]}) [Done]')

    return X_tr, X_te, features_names
