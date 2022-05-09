# Topic-agnostic features for authorship analysis in Spanish political speeches

## Abstract
<p>Among the many tasks of the authorship field, Authorship Identification (AId) aims at uncovering the author of a document, while Author Profiling (AP) focuses on the analysis of personal characteristics of the author(s), such as gender, age, etc. Methods devised for such tasks typically focus on the <em>style</em> of the writing, and are expected not to make inferences grounded on the <em>topics</em> that certain authors tend to write about. <br>
We present a series of experiments evaluating the use of topic-agnostic feature sets for Authorship Identification and Author Profiling tasks in Spanish political language. In particular, we propose to employ features based on rhythmic and psycholinguistic patterns, obtained via different approaches of text masking that we use to actively mask the underlying topic. We feed these feature sets to a SVM learner, and show that they lead to results that are comparable to 
those obtained by a BETO transformer, when the latter is trained on the original text, i.e., potentially learning from topical information. Moreover, we further investigate the results for the different authors, showing that variations in performance are partially explainable in terms of the authors' political affiliation and communication style. <br>
These experiments were presented on two articles: [[1]](#1) (preliminary) and [[2]](#2) .</p>

## Topic-agnostic features for authorship tasks
# Topic-agnostic features
<p>We explore various combinations of topic-agnostic feature sets. In particular, we focus on features extracted from syllabic stress and LIWC [[3]](#3) categories.
<ul>
<li><strong>base_features</strong>: widely-used feature from the literature. The set is comprised of:
<ul>
<li>relative frequencies of function words (from the [NLTK library](https://www.nltk.org/))</li>
<li>relative frequencies word lengths</li>
<li>relative frequencies sentence lengths</li>
</ul></li>
<li><strong>pos_tags</strong>: we replace each word in the document with the respective Part-of-Speech tag; we then extract the word n-grams in the range [1,3] and compute the TfIdf weights </li>
<li><strong>stress</strong>: we convert the document into a sequence of stressed and unstressed syllables (using the output of the [Rantanplan library](https://github.com/linhd-postdata/rantanplan)); we then extract the character n-grams in the range [1,7] and compute the TfIdf weights. </li>
<li><strong>LIWC_GRAM</strong>: we replace each word in the document with the respective LIWC category tag (representing grammatical information); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights.</li>
<li><strong>LIWC_OBJ</strong>: we replace each word in the document with the respective LIWC category tag (representing objects and concepts); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights. <strong>Not used in the project.</strong></li>
<li><strong>LIWC_COG</strong>: we replace each word in the document with the respective LIWC category tag (representing cognitive processes or actions); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights.</li>
<li><strong>LIWC_FEELS</strong>: we replace each word in the document with the respective LIWC category tag (representing feelings and emotions); we then extract the word n-grams in the range [1,3] and compute the TfIdf weights.</li>
</ul>

In the code, the features to use are instructed via a dictionary of <em>feature_set:boolean</em> passed from <em>main.py</em>. E.g., <em>{'base_features': True, 'pos_tags': True, 'stress': False, 'liwc_gram': False, 'liwc_obj': False,
             'liwc_cog': False, 'liwc_feels': False}</em> extracts and uses only the base_features and pos_tags features sets.<br>
We apply a feature selection approach on the features derived from the LIWC encodings: we keep only the 10% most important features for each LIWC encodings. This can be changed in <em>feature_extractor.py</em>.</p>

# Authorship tasks
<p>We perform classification experiments of AId and AP in various settings:
<ul>
<li>For AId, we tackle the tasks of Authorship Attribution (AA) and Authorship Verification (AV)
<ul>
<li>AA: each sample is labelled as belonging to one of the authors in the dataset</li>
<li>AV: for each author in the dataset, each sample is labelled as belonging to that author or not</li></ul></li>
<li> For AP, we tackle various tasks where each sample is labelled as belonging to the gender, age group, political wing or political party of its author
</ul>
In the code, the desired setting can be selected via two parameters in <em>main.py</em>:
<ul>
<li><em>task</em>: can be BC (binary classification, only 2 classes) or MC (multiclass classification, more than 2 classes) </li>
<li><em>focus</em>: can be name, gender, birth (age group), wing, party, party_status (coalition or opposition) </li>
</ul>
E.g., in order to perform AV, the input is <em>task = BC</em> and <em>focus = name</em>; in order to perform AP by gender, the input is <em>task = MC</em> and <em>focus = gender</em>. 
</p>

# Dataset
<p>We employ the Spanish repository (covering the years 2015-2020) of the [Linguistically annotated multilingual comparable corpora of parliamentary debates ParlaMint.ana 2.1](https://www.clarin.si/repository/xmlui/handle/11356/1431) which contains the annotated transcriptions of many sessions of various European Parliaments. Because of their declamatory nature, between the written text and the discourse, these speeches seem particularly suited for an investigation on rhythm and psycholinguistic traits.</p>

# Classic ML algorithms and BETO
<p>Regarding the learning algorithm, our work focuses on Support Vector Machine (SVM), but also Random Forest (RF) and Logistic Regression (LR) can be selected. We employ the implementations from the [scikit-learn package](https://scikit-learn.org/stable/), and we perform the optimization of various hyper-parameters in a grid-search fashion, via 5-fold cross-validation on the training set. <br>
In order to compare our result, we employ the pre-trained transformer named BETO-cased, from the [Hugginface library] (https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) on the original text (hence, potentially using topic-related information). We fine-tune the model for 50 epochs on the training set before use. <br>
In the code, the desired learning algorithm can be selected via the parameter <em>learner_name</em>.</p>

## Post-hoc analysis of the AV results
<p> Given the differences in performance among authors spotted in the AV results, we perform further analysis in order to seek a possible explanation. These can be found in <em>explainability.py</em>. </p>

# One-way ANOVA test
<p> We use the one-way ANOVA in order to see if, by grouping the results of the speakers by categories (wing, party, gender, or age group), statistically significant differences emerge among the groups.<br>
We employ the [SciPy](https://scipy.org/) and [statsmodels](https://www.statsmodels.org/stable/index.html) libraries. </p>

# Spearman coefficient to style indices
<p> We compute 3 style indices:
<ul>
<li>the Analytic Thinking Index (ATI)</li>
<li>the Categorical-versus-Narrative Index (CNI)</li>
<li>the Adversarial Style Index (ASI)</li>
</ul>
We employ these measures to quantify the extent to which the AV performance correlates to certain styles of communication.<br>
To do so, we compute the Spearman correlation coefficient between the classification scores and the authors' index scores, employing the [SciPy library](https://scipy.org/). </p>

## Code 
<p>The code is organized as follows int the <strong>src</strong> directory:
<ul>
<li><strong>general</strong>: the directory contains: 
<ul>
<li> <em>helpers.py</em>: various functions useful for the current project </li>
<li> <em>LIWC.py</em>: code for reading the LIWC dictionary, defining the LIWC macro-categories (GRAM, OBJ, COG, FEELS) and encode the text </li>
<li> <em>sign_test.py</em>: code for the significance test </li>
<li> <em>utils.py</em>: various functions potentially useful for multiple projects </li>
<li> <em>visualization.py</em>: code for plots and graphics </li>
</ul>
<li><em>explainability.py</em>: various functions to analyze the AV results </li>
<li><em>feature_extractor.py</em>: code for features extraction with the desired feature sets </li>
<li><em>main.py</em>
<li><em>ml_classification.py</em>: code for classification tasks with classic ML algorithms </li>
<li><em>process_dataset.py</em>: functions to select and process the speeches in various ways </li>
<li><em>trans_classification.py</em>: code for classification tasks with BETO </li>
</ul>
</p>


## References
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
