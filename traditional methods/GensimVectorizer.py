from sklearn.base import BaseEstimator, TransformerMixin
import gensim
from gensim.models.fasttext import FastText

import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors



class GensimVectorizer(BaseEstimator, TransformerMixin):
    # initialize all parameters for word2vec and fasttext

    def __init__(self, unlab, model_type, own_domain=True, finetune=False, size=300, alpha=0.025, window=5, min_count=4,
                 max_vocab_size=None,
                 sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5,
                 ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000,
                 callbacks=(), max_final_vocab=None):

        self.unlab = unlab
        self.model_type = model_type
        self.own_domain = own_domain
        self.finetune = finetune
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = max_final_vocab

    def fit(self, X, y=None):

        # train on task-specific domain
        if self.own_domain == True:
            if self.model_type == 'Word2Vec':
                self.model_ = Word2Vec(self.unlab, corpus_file=None,
                                       size=self.size, alpha=self.alpha, window=self.window, min_count=self.min_count,
                                       max_vocab_size=self.max_vocab_size, sample=self.sample, seed=self.seed,
                                       workers=self.workers, min_alpha=self.min_alpha, sg=self.sg, hs=self.hs,
                                       negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
                                       hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word,
                                       trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab,
                                       batch_words=self.batch_words,
                                       callbacks=self.callbacks,
                                       max_final_vocab=self.max_final_vocab)

                self.model_.build_vocab(X, update=True)
                self.model_.train(X, total_examples=3, epochs=1)
            if self.model_type == 'FastText':
                self.model_ = FastText(X, corpus_file=None,
                    size=self.size, alpha=self.alpha, window=self.window, min_count=self.min_count,
                    max_vocab_size=self.max_vocab_size, sample=self.sample, seed=self.seed,
                    workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
                    negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
                    hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word,
                    trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words, callbacks=self.callbacks,
                    max_final_vocab=self.max_final_vocab)
            return self

        # train on general domain
        if self.own_domain == False:
            # WORD2VEC
            if self.model_type == 'Word2Vec':
                # fine-tuning
                if self.finetune == True:
                    self.model_ = gensim.models.Word2Vec(size=300, window=5, min_count=2)

                    self.model_.build_vocab(X)
                    self.model_.build_vocab(self.unlab)

                    self.model_.intersect_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', lockf=1.0,
                                                          binary=True)

                    self.model_.train(self.unlab, total_examples=3, epochs=self.iter)
                    self.model_.train(X, total_examples=3, epochs=self.iter)

                # feature-based
                if self.finetune == False:
                    self.model_ = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz',
                                                                    binary=True)
            # FASTTEXT
            if self.model_type == 'FastText':
                self.model_ = KeyedVectors.load_word2vec_format('../data/wiki-news-300d-1M.vec', limit=999999)
                self.model_.build_vocab(X, update=True)
                self.model_.train(sentences=X, total_examples = len(X), epochs=5)

            return self


    def transform(self, X):
        X_embeddings = np.array([self._get_embedding(words) for words in X])
        return X_embeddings

    def _get_embedding(self, words):
        '''
        Average all embeddings before feeding to the classifier
        '''

        valid_words = [word for word in words if word in self.model_.wv.vocab]
        if valid_words:
            embedding = np.zeros((len(valid_words), self.size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                embedding[idx] = self.model_.wv[word]

            return np.mean(embedding, axis=0)
        else:
            return np.zeros(self.size)