import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from GensimVectorizer import *

from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gensim

#import scikitplot.plotters as skplt

import nltk

# from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

# HER
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from multiprocessing import Process, freeze_support


# HOUDEN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression


# load unlabeled data
df_unlabeled = pd.read_csv('../data/label_prediction_logging.csv', sep=',')
df_unlabeled = df_unlabeled[['input_text']]
df_unlabeled.columns = ['text']

# load labeled data
df_labeled = pd.read_csv('../data/27-03.csv', header=None, sep=';')
df_labeled.columns = ['text', 'label']


def plot_confusion_matirx(cm, labels):
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



def imbalanced_dataset(train_set):
    train_set.reset_index(inplace=True)
    train_set.reset_index(inplace=True)
    train_set.columns = ['id', 'id1', 'text', 'label']
    train_set.drop(columns=['id1'], inplace=True)

    labels = train_set.groupby('label').id.unique()

    # sort the over-represented class
    labels = labels[labels.apply(len).sort_values(ascending=False).index]

    temp = [len(labels.iloc[0]), len(labels.iloc[1]), len(labels.iloc[2])]

    excess0 = len(labels.iloc[0]) - len(labels.iloc[2])
    excess1 = len(labels.iloc[1]) - len(labels.iloc[2])

    remove0 = np.random.choice(labels.iloc[0], excess0, replace=False)
    remove1 = np.random.choice(labels.iloc[1], excess1, replace=False)
    joined_list = [*remove0, *remove1]

    df_balanced_train = train_set[~train_set.index.isin(joined_list)].copy()
    df_balanced_train.drop(columns=['id'], inplace=True)


def clean(data):
    data['text'] = data['text'].replace(r'\n', ' ', regex=True)  # inplace=True
    data['text'] = data['text'].replace(r'\d+', ' ', regex=True)
    data['text'] = data['text'].replace(r'[^\w\s]', ' ', regex=True)
    data['text'] = data['text'].replace(r'\_', ' ', regex=True)
    data['text'] = data['text'].replace(r'\b[a-zA-Z]\b', ' ', regex=True)
    data['text'] = data['text'].replace(r'\s+', ' ', regex=True)
    data['text'] = data['text'].replace(r'^\s+', '', regex=True)
    return data



def visualize():



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
        w2v = GensimVectorizer(unlab, model_type='Word2Vec', own_domain=own_domain, finetune=finetune, sg=0)  # , callbacks=[callback()]
        print(w2v)

        pipeline = Pipeline([
            ("word2vec", w2v),
            ("classifier", LogReg)
        ])

        param_grid = {}
        # 'classifier__C': [1, 10, 100, 1000], 'word2vec__alpha': [0.1, 0.01, 0.02, 0.001], 'word2vec__window': [4,5,6,7], 'word2vec__iter': [5, 10, 15, 20]}

        return pipeline, param_grid

    if method == 'fasttext':
        ft = GensimVectorizer(unlab, model_type='FastText', own_domain=True, finetune=True, sg=0)  # , callbacks=[callback()]
        pipeline = Pipeline([
            ("fasttext", ft),
            ("classifier", LogReg)
        ])
        param_grid = {'classifier__C': [1, 10, 100, 1000], 'fasttext__alpha': [0.1, 0.01, 0.02, 0.001], 'fasttext__window': [4,5,6,7], 'fasttext__iter': [5, 10, 15, 20]}

        return pipeline, param_grid



def main():
    freeze_support()

    # clean labeled and unlabeled dataset
    df_labeled_ = clean(df_labeled)
    df_unlabeled_ = clean(df_unlabeled)

    # split in train and test
    train_set, test_set = train_test_split(df_labeled_, test_size=0.2, random_state=42)

    # initialize method: 'tfidf', 'word2vec', 'fasttext'
    method = 'word2vec'
    # task-specific domain=True, general domain=False
    domain = False
    # fine-tune=True, feature-based=False
    finetune = False

    # create pipeline and hyperparameter set
    pipeline_, param_grid = pipeline(method, domain, finetune, df_unlabeled_)
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






#balanced
# print("BEST PARAMS", log_grid.best_params_)
# print(classification_report(log_grid.best_estimator_.predict(test_set['text']), test_set['label']))

#
#
#
# # phrases takes a list of words as input
# data_words = [row.split() for row in df['text']]
# # Build the bigram and trigram models
# bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
#
# # Faster way to get a sentence clubbed as a trigram/bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)
#
# def w2v_tokenize_text(text):
#     tokens = []
#     for sent in nltk.sent_tokenize(text, language='english'):
#         for word in nltk.word_tokenize(sent, language='english'):
#             if len(word) < 2:
#                 continue
#             tokens.append(word)
#     return tokens
#
#
# df['label'] = df['label'].astype(int)
# df['text'] = df['text'].astype(str)
# # tokenize text
# train_text = train.apply(lambda r: w2v_tokenize_text(r['text']), axis=1).values
# test_text = test.apply(lambda r: w2v_tokenize_text(r['text']), axis=1).values
#
# # labels
# train_labels = np.array([label for label in train['label']])
# test_labels = np.array([label for label in test['label']])
#
#
#
# class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         self.dim = len(next(iter(word2vec.values())))
#         print(self.dim)
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X):
#         return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec]
#                                  or [np.zeros(self.dim)], axis=0) for words in X])
#
#
#
# # GRIDSEARCH
# param_grid = {'logreg__C': [1, 10, 100, 1000, 1e4, 1e5] #1e5
# }
#
#
#
# # WORD2VEC
# # model_wtv
# model = gensim.models.Word2Vec(trigram_mod[bigram_mod[data_words]], size=300, window=5, min_count=2)
# #model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', limit=50000, binary=False)
# # model_w2v_pt
# # # FastText
# model = gensim.models.FastText(trigram_mod[bigram_mod[data_words]], size=100, window=5, min_count=2)
# # model = KeyedVectors.load_word2vec_format('./data/wiki-news-300d-1M.vec', limit=99999)
# #
# # models = [model_wtv, model_w2v_pt, model_ft, model_ft_pt]
#
#
#
#
#
# def main():
#     freeze_support()
#     model_dic = dict(zip(model.wv.index2word, model.wv.syn0))
#
#
#     pipeline = Pipeline([
#         ("word2vec vectorizer", MeanEmbeddingVectorizer(model_dic)),
#         ("logreg", LogisticRegression(random_state=42, multi_class='multinomial'))])
#
#     log_grid = GridSearchCV(pipeline, param_grid=param_grid, scoring="accuracy", verbose=3, cv=5, n_jobs=1)
#     log_grid.fit(train_text, train_labels) # fitted =
#
#     print("BEST PARAMS", log_grid.best_params_)
#
#     print(classification_report(log_grid.best_estimator_.predict(test_text), test_labels))
#
#     predictions = log_grid.best_estimator_.predict(test_text)
#     y_test = test['label'].tolist()
#     # # PLOT CONFUSION MATRIX
#     conf_mat = metrics.confusion_matrix(y_test, predictions)
#     fig, ax = plot_confusion_matirx(conf_mat, ['other', 'reminder', 'invoice'])
#     fig.savefig('./plots/w2v.png')
#     plt.close()
#
#     with open("file.txt", "a") as output:
#         output.write("---------------------------------------------------------------")
#         output.write(str("\n"))
#         output.write(str(model))
#         output.write(str("\n"))
#         output.write(str(conf_mat))
#         output.write(str("\n"))
#         output.write(str(log_grid.best_params_))
#         output.write(str("\n"))
#         output.write(classification_report(log_grid.best_estimator_.predict(test_text), test_labels))
#         output.write("---------------------------------------------------------------")
#         output.write("\n\n")
#
#     #----------------------------------
#
#     keys = ['Paris', 'Python', 'Sunday', 'Tolstoy', 'Twitter', 'bachelor', 'delivery', 'election', 'expensive',
#             'experience', 'financial', 'food', 'iOS', 'peace', 'release', 'war']
#
#     embedding_clusters = []
#     word_clusters = []
#     for word in keys:
#         embeddings = []
#         words = []
#         for similar_word, _ in model.most_similar(word, topn=30):
#             words.append(similar_word)
#             embeddings.append(model[similar_word])
#         embedding_clusters.append(embeddings)
#         word_clusters.append(words)
#
#     #
#     # from sklearn.manifold import TSNE
#     # import numpy as np
#     #
#     # embedding_clusters = np.array(embedding_clusters)
#     # n, m, k = embedding_clusters.shape
#     # tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
#     # embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
#     #
#     #
#     # def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
#     #     plt.figure(figsize=(16, 9))
#     #     colors = cm.rainbow(np.linspace(0, 1, len(labels)))
#     #     for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
#     #         x = embeddings[:, 0]
#     #         y = embeddings[:, 1]
#     #         plt.scatter(x, y, c=color, alpha=a, label=label)
#     #         for i, word in enumerate(words):
#     #             plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
#     #                          textcoords='offset points', ha='right', va='bottom', size=8)
#     #     plt.legend(loc=4)
#     #     plt.title(title)
#     #     plt.grid(True)
#     #     if filename:
#     #         plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
#     #     plt.show()
#     #
#     #
#     # tsne_plot_similar_words('Similar words from Google News', keys, embeddings_en_2d, word_clusters, 0.7,
#     #                         'similar_words.png')



if __name__ == "__main__":
    main()