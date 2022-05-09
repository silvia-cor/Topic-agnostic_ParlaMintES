# Topic-agnostic features for authorship analysis in Spanish political speeches

## Abstract
Among the many tasks of the authorship field, Authorship Identification (AId) aims at uncovering the author of a document, while Author Profiling (AP) focuses on the analysis of personal characteristics of the author(s), such as gender, age, etc. Methods devised for such tasks typically focus on the *style* of the writing, and are expected not to make inferences grounded on the *topics* that certain authors tend to write about. 

We present a series of experiments evaluating the use of topic-agnostic feature sets for Authorship Identification and Author Profiling tasks in Spanish political language. In particular, we propose to employ features based on rhythmic and psycholinguistic patterns, obtained via different approaches of text masking that we use to actively mask the underlying topic. 
We feed these feature sets to a SVM learner, and show that they lead to results that are comparable to those obtained by a BETO transformer, when the latter is trained on the original text, i.e., potentially learning from topical information. 
Moreover, we further investigate the results for the different authors, showing that variations in performance are partially explainable in terms of the authors' political affiliation and communication style. 
These experiments were presented in two articles: [[1]](#1) (preliminary) and [[2]](#2) .

# Topic-agnostic features for authorship tasks
## Topic-agnostic features
We explore various combinations of topic-agnostic feature sets. In particular, we focus on features extracted from syllabic stress and LIWC [[3]](#3) categories.

- **base_features**: widely-used feature from the literature. The set is comprised of:
    - relative frequencies of function words (from the [NLTK library](https://www.nltk.org/))
    - relative frequencies word lengths
    - relative frequencies sentence lengths
- **pos_tags**: we replace each word in the document with the respective Part-of-Speech tag; we then extract the word n-grams in the range [1,3] and compute the TfIdf weights 
- **stress**: we convert the document into a sequence of stressed and unstressed syllables (using the output of the [Rantanplan library](https://github.com/linhd-postdata/rantanplan)); we then extract the character n-grams in the range [1,7] and compute the TfIdf weights. 
- **LIWC_GRAM**: we replace each word in the document with the respective LIWC category tag (representing grammatical information); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights.
- **LIWC_OBJ**: we replace each word in the document with the respective LIWC category tag (representing objects and concepts); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights. **Not used in the project.**
- **LIWC_COG**: we replace each word in the document with the respective LIWC category tag (representing cognitive processes or actions); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights.
- **LIWC_FEELS**: we replace each word in the document with the respective LIWC category tag (representing feelings and emotions); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights.

In the code, the features to use are instructed via a dictionary of `feature_set:boolean` passed from `main.py`. E.g., ``{'base_features': True, 'pos_tags': True, 'stress': False, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': False}`` extracts and uses only the base_features and pos_tags features sets.

We apply a feature selection approach on the features derived from the LIWC encodings: we keep only the 10% most important features for each LIWC encodings. This can be changed in `feature_extractor.py`.

## Authorship tasks
We perform classification experiments of AId and AP in various settings:

- For AId, we tackle the tasks of Authorship Attribution (AA) and Authorship Verification (AV)
    - AA: each sample is labelled as belonging to one of the authors in the dataset
    - AV: for each author in the dataset, each sample is labelled as belonging to that author or not
- For AP, we tackle various tasks where each sample is labelled as belonging to the gender, age group, political wing or political party of its author

In the code, the desired setting can be selected via two parameters in `main.py`:

- `task`: can be BC (binary classification, only 2 classes) or MC (multiclass classification, more than 2 classes) 
- `focus`: can be name, gender, birth (age group), wing, party, party_status (coalition or opposition) 

E.g., in order to perform AV, the input is `task = BC` and `focus = name`; in order to perform AP by gender, the input is `task = MC` and `focus = gender`. 


## Dataset
We employ the Spanish repository (covering the years 2015-2020) of the [Linguistically annotated multilingual comparable corpora of parliamentary debates ParlaMint.ana 2.1](https://www.clarin.si/repository/xmlui/handle/11356/1431) which contains the annotated transcriptions of many sessions of various European Parliaments. Because of their declamatory nature, between the written text and the discourse, these speeches seem particularly suited for an investigation on rhythm and psycholinguistic traits.

## Classic ML algorithms and BETO
Regarding the learning algorithm, our work focuses on Support Vector Machine (SVM), but also Random Forest (RF) and Logistic Regression (LR) can be selected. We employ the implementations from the [scikit-learn package](https://scikit-learn.org/stable/), and we perform the optimization of various hyper-parameters in a grid-search fashion, via 5-fold cross-validation on the training set. 

In order to compare our result, we employ the pre-trained transformer named BETO-cased, from the [Hugginface library] (https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) on the original text (hence, potentially using topic-related information). We fine-tune the model for 50 epochs on the training set before use. 

In the code, the desired learning algorithm can be selected via the parameter `learner_name`.

# Post-hoc analysis of the AV results
 Given the differences in performance among authors spotted in the AV results, we perform further analysis in order to seek a possible explanation. These can be found in `explainability.py`. 

## One-way ANOVA test
We use the one-way ANOVA in order to see if, by grouping the results of the speakers by categories (wing, party, gender, or age group), statistically significant differences emerge among the groups.

We employ the [SciPy](https://scipy.org/) and [statsmodels](https://www.statsmodels.org/stable/index.html) libraries. 

## Spearman coefficient to style indices
We compute 3 style indices:

- the Analytic Thinking Index (ATI)
- the Categorical-versus-Narrative Index (CNI)
- the Adversarial Style Index (ASI)

We employ these measures to quantify the extent to which the AV performance correlates to certain styles of communication.
To do so, we compute the Spearman correlation coefficient between the classification scores and the authors' index scores, employing the [SciPy library](https://scipy.org/). 

### Code 
The code is organized as follows int the `src` directory:

- `general`: the directory contains: 
    - `helpers.py`: various functions useful for the current project 
    - `LIWC.py`: code for reading the LIWC dictionary, defining the LIWC macro-categories (GRAM, OBJ, COG, FEELS) and encode the text 
    - `sign_test.py`: code for the significance test 
    - `utils.py`: various functions potentially useful for multiple projects 
    - `visualization.py`: code for plots and graphics 
- `explainability.py`: various functions to analyze the AV results 
- `feature_extractor.py`: code for features extraction with the desired feature sets 
- `main.py`
- `ml_classification.py`: code for classification tasks with classic ML algorithms 
- `process_dataset.py`: functions to select and process the speeches in various ways 
- `trans_classification.py`: code for classification tasks with BETO 


### References
<a id="1">[1]</a>
Corbara, S., Chulvi, B., Rosso, P., Moreo, A. (2022) 
Investigating topic-agnostic features for authorship tasks in Spanish political speeches. 
In: Proceedings of the 27th International Conference on Applications of Natural Language to Information Systems (NLDB 2022), Springer.

<a id="2">[2]</a>
Corbara, S., Chulvi, B., Rosso, P., Moreo, A. (2022) 
Rythmic and psycolinguistic features for authorship tasks in the Spanish Parliament: Evaluation and analysis. 
(submitted)

<a id="3">[3]</a>
Pennebaker, J.W., Boyd, R.L., Jordan, K., Blackburn, K. (2015) 
The development and psychometric properties of LIWC2015. Technical report. 
