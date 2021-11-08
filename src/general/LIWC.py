import sys
import re
from operator import itemgetter


# Function to read a LIWC-style dictionary
# return a dictionary for each of the listed macro-categories
def read_LIWC_dict(dict_path='../dataset/Spanish_LIWC2007_Dictionary.dic'):
    # Macro-categories (grammar, object/concept, cognitive process, feeling)
    cat_gram = ['PronPer', 'PronImp', 'Yo', 'Nosotro', 'TuUtd', 'ElElla', 'VosUtds', 'Ellos', 'Pasado',
                'Present', 'Futuro', 'Subjuntiv', 'Negacio', 'Cuantif', 'Numeros',
                'verbYO', 'verbTU', 'verbNOS', 'verbVos', 'verbosEL', 'verbELLOS', 'formal', 'informal']
    cat_obj = ['Social', 'Familia', 'Amigos', 'Humanos',
               'Biolog', 'Cuerpo', 'Salud', 'Sexual', 'Espacio', 'Tiempo',
               'Trabajo', 'Logro', 'Hogar', 'Dinero', 'Relig', 'Muerte']
    cat_cog = ['MecCog', 'Insight', 'Causa', 'Discrep', 'Tentat', 'Certeza', 'Inhib', 'Incl', 'Excl',
               'Percept', 'Ver', 'Oir', 'Sentir', 'NoFluen', 'Relleno', 'Ingerir', 'Relativ', 'Movim']
    cat_feels = ['Maldec', 'Afect', 'EmoPos', 'EmoNeg', 'Ansiedad', 'Enfado', 'Triste', 'Asentir', 'Placer']
    cat_location = []
    dic = {}
    # check to make sure the dictionary is properly formatted
    with open(dict_path, "r") as dict_file:
        for idx, item in enumerate(dict_file):
            if "%" in item:
                cat_location.append(idx)
        if len(cat_location) > 2:
            # There are apparently more than two category sections; throw error and exit
            sys.exit("Invalid dictionary format. Check the number/locations of the category delimiters (%).")
    # read dictionary as lines
    with open(dict_path, "r") as dict_file:
        lines = dict_file .readlines()
    # for each macro-category create a list of tuples (number, category)
    splits = []
    for line in lines[cat_location[0] + 1:cat_location[1]]:
        splits.append(tuple(re.split(r'\t+', line.replace('\n', ''))))
    cat_gram = [split for split in splits if split[1] in cat_gram]
    cat_obj = [split for split in splits if split[1] in cat_obj]
    cat_cog = [split for split in splits if split[1] in cat_cog]
    cat_feels = [split for split in splits if split[1] in cat_feels]
    # create a dictionary with every word and its categories
    for line in lines[cat_location[1] + 1:]:
        row = re.split('\t', line.rstrip())
        dic[row[0]] = list(row[1:])
    # create a dictionary for each macro-category
    # TODO: better encoding?
    dic_gram = {}
    dic_obj = {}
    dic_cog = {}
    dic_feels = {}
    for word in dic:
        dic_gram[word] = _assign_word_cat(word, cat_gram, dic)
        dic_obj[word] = _assign_word_cat(word, cat_obj, dic)
        dic_cog[word] = _assign_word_cat(word, cat_cog, dic)
        dic_feels[word] = _assign_word_cat(word, cat_feels, dic)
    dic_liwc = {'dic_gram': dic_gram, 'dic_obj': dic_obj, 'dic_cog': dic_cog, 'dic_feels': dic_feels}
    return dic_liwc


# assign an encoding to each word
# every word is encoded as the union (joined strings) of its category (given a macro-category)
# if a word has more than one category, they are joined with a '*' symbol, e.g. '113*27'
# if a word does not have a category, it is encoded as '0'
def _assign_word_cat(word, macro_cat, dic):
    res = '*'.join(cat for cat in dic[word] if cat in map(itemgetter(0), macro_cat))
    if res != '':
        return res
    else:
        return '0'
