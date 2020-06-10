import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from GensimVectorizer import *

from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from multiprocessing import Process, freeze_support
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression


# load unlabeled data
df_unlabeled = pd.read_csv('../data/label_prediction_logging.csv', sep=',')
df_unlabeled = df_unlabeled[['input_text']]
df_unlabeled.columns = ['text']

df_unlabeled['text'] = df_unlabeled['text'].astype(str)
data_words = [row.split() for row in df_unlabeled['text']]

# create bigram and trigram using gensim
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# final unlabeled dataset containing bigrams and trigrams
df_unlab = trigram_mod[bigram_mod[data_words]]

# load labeled data
df_labeled = pd.read_csv('../data/27-03.csv', header=None, sep=';')
df_labeled.columns = ['text', 'label']




def plot_confusion_matirx(cm, labels):
    '''
    Function for plotting confusion matrix
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues', origin='upper')

    ax.figure.colorbar(im, ax=ax)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title="confusion matrix",
           ylabel='True label',
           xlabel='Predicted label')
    plt.axis(xmin=-0.5, xmax=np.shape(cm)[0] - 0.5, ymin=np.shape(cm)[0] - 0.5, ymax=-0.5)

    return fig, ax



# def balanced_dataset(train_set):
#     '''
#     Function for creating balanced dataset and removes instances of over-represented classes
#     '''
#
#     train_set.reset_index(inplace=True)
#     train_set.reset_index(inplace=True)
#     train_set.columns = ['id', 'id1', 'text', 'label']
#     train_set.drop(columns=['id1'], inplace=True)
#
#     labels = train_set.groupby('label').id.unique()
#
#     # sort the over-represented class
#     labels = labels[labels.apply(len).sort_values(ascending=False).index]
#
#     temp = [len(labels.iloc[0]), len(labels.iloc[1]), len(labels.iloc[2])]
#
#     excess0 = len(labels.iloc[0]) - len(labels.iloc[2])
#     excess1 = len(labels.iloc[1]) - len(labels.iloc[2])
#
#     remove0 = np.random.choice(labels.iloc[0], excess0, replace=False)
#     remove1 = np.random.choice(labels.iloc[1], excess1, replace=False)
#     joined_list = [*remove0, *remove1]
#
#     df_balanced_train = train_set[~train_set.index.isin(joined_list)].copy()
#     df_balanced_train.drop(columns=['id'], inplace=True)


def clean(data):
    '''
    Cleaning text from symbols
    '''
    data['text'] = data['text'].replace(r'\n', ' ', regex=True)  # inplace=True
    data['text'] = data['text'].replace(r'\d+', ' ', regex=True)
    data['text'] = data['text'].replace(r'[^\w\s]', ' ', regex=True)
    data['text'] = data['text'].replace(r'\_', ' ', regex=True)
    data['text'] = data['text'].replace(r'\b[a-zA-Z]\b', ' ', regex=True)
    data['text'] = data['text'].replace(r'\s+', ' ', regex=True)
    data['text'] = data['text'].replace(r'^\s+', '', regex=True)
    return data




def pipeline(method, own_domain=None, finetune=None, unlab=None):

    '''
    Set different pipelines for tf-idf, word2vec and fasttext
    '''

    # classifier
    LogReg = LogisticRegression(random_state=42, multi_class='multinomial', solver='newton-cg')

    if method == 'tfidf':

        pipeline = Pipeline([
            ('bow', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', LogReg)
        ])

        param_grid = {'classifier__C': [1, 10, 100, 1000]}

        return pipeline, param_grid

    if method == 'word2vec':
        w2v = GensimVectorizer(unlab, model_type='Word2Vec', own_domain=own_domain, finetune=finetune, sg=0)

        pipeline = Pipeline([
            ("word2vec", w2v),
            ("classifier", LogReg)
        ])

        param_grid = {'classifier__C': [1, 10, 100, 1000], 'word2vec__alpha': [0.1, 0.01, 0.02, 0.001], 'word2vec__window': [4,5,6,7], 'word2vec__iter': [5, 10, 15, 20]}

        return pipeline, param_grid

    if method == 'fasttext':
        ft = GensimVectorizer(unlab, model_type='FastText', own_domain=True, finetune=True, sg=0)
        pipeline = Pipeline([
            ("fasttext", ft),
            ("classifier", LogReg)
        ])
        param_grid = {'classifier__C': [1, 10, 100, 1000], 'fasttext__alpha': [0.1, 0.01, 0.02, 0.001], 'fasttext__window': [4,5,6,7], 'fasttext__iter': [5, 10, 15, 20]}

        return pipeline, param_grid
    else:
        return 0, 0



def main():
    freeze_support()

    # clean labeled and unlabeled dataset
    df_labeled_ = clean(df_labeled)

    # split in train and test
    train_set, test_set = train_test_split(df_labeled_, test_size=0.2, random_state=42)

    # initialize method: 'tfidf', 'word2vec', 'fasttext'
    method = 'word2vec'
    # task-specific domain=True, general domain=False
    domain = True
    # fine-tune=True, feature-based=False
    finetune = False

    # create pipeline and hyperparameter set
    pipeline_, param_grid = pipeline(method, domain, finetune, df_unlab)
    log_grid = GridSearchCV(pipeline_, param_grid=param_grid, scoring="accuracy", verbose=3, cv=5, n_jobs=1)

    # train representation method + classifier
    log_grid.fit(train_set['text'], train_set['label'])
    # y_score_tfidf = log_grid.predict_proba(test_set['text'])

    # print best performing parameters and report
    print("BEST PARAMS", log_grid.best_params_)
    print(classification_report(log_grid.best_estimator_.predict(test_set['text']), test_set['label']))

    predictions = log_grid.best_estimator_.predict(test_set['text'])

    # plot confusion matrix
    conf_mat = metrics.confusion_matrix(test_set['label'], predictions)
    j, y = plot_confusion_matirx(conf_mat, ['other', 'reminder', 'invoice'])


if __name__ == "__main__":
    main()