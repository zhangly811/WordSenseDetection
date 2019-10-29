from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
import scipy.sparse
from six.moves import cPickle as pickle
from six.moves import urllib
import tensorflow as tf
import tensorflow_probability as tfp

def download(directory, filename):
  """Download a file."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  # url = os.path.join(ROOT_PATH, filename)
  # print("Downloading %s to %s" % (url, filepath))
  # urllib.request.urlretrieve(url, filepath)
  return filepath


def PubMed_dataset(directory, split_name, num_words):
  """Return tf.data.Dataset."""
  data = np.load(download(directory, FILE_TEMPLATE.format(split=split_name)), allow_pickle=True, encoding="bytes")
  # The last row is empty in both train and test.
  # data = data[:-1]

  # Each row is a list of word ids in the document. We first convert this to
  # sparse COO matrix (which automatically sums the repeating words). Then,
  # we convert this COO matrix to CSR format which allows for fast querying of
  # documents.
  num_documents = data.shape[0]
  indices = np.array([(row_idx, column_idx)
                      for row_idx, row in enumerate(data)
                      for column_idx in row])
  sparse_matrix = scipy.sparse.coo_matrix(
      (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
      shape=(num_documents, num_words),
      dtype=np.float32)
  sparse_matrix = sparse_matrix.tocsr()
  return sparse_matrix

directory = "/Users/linyingzhang/git/zhangly811/WordSenseDetection/dat_unsync/data"
FILE_TEMPLATE = "{split}.txt.npy"

with open(download(directory, "vocab.pkl"), "rb") as f:
    words_to_idx = pickle.load(f)
num_words = len(words_to_idx)

vocabulary = [None] * num_words
for word, idx in words_to_idx.items():
    vocabulary[idx] = word

sparse_matrix = PubMed_dataset(directory, 'train', num_words)
X = sparse_matrix.todense()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(X.T)
toppcs=pca.components_[:7]
print(pca.explained_variance_)



#### tSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_2d = tsne.fit_transform(toppcs.T)
y = np.array([x.strip() for x in open("/Users/linyingzhang/git/zhangly811/WordSenseDetection/dat_unsync/data/cold_train.labels", "rt")])
y[y=="M1"]=0
y[y=="M2"]=1
y[y=="M3"]=2
y = np.array(y, dtype=int)
target_names = ['Cold Temp', 'COLD', 'the Common Cold']
target_ids = [0,1,2]

plt.figure(figsize=(6, 5))
colors = 'r','b','y'
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()


from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=3, random_state=0).fit(toppcs.T)
kmeans.labels_

bool = [y[i]==kmeans.labels_[i] for i in range(y.shape[0])]

acc = sum(bool)/len(bool)
acc
##################################
# Natural Language Toolkit: Word Sense Disambiguation Algorithms
#
# Authors: Liling Tan <alvations@gmail.com>,
#          Dmitrijs Milajevs <dimazest@gmail.com>
#
# Copyright (C) 2001-2019 NLTK Project
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from nltk.corpus import wordnet


def lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    """Return a synset for an ambiguous word in a context.

    :param iter context_sentence: The context sentence where the ambiguous word
         occurs, passed as an iterable of words.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.

    This function is an implementation of the original Lesk algorithm (1986) [1].

    Usage example::

        >>> lesk(['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.'], 'bank', 'n')
        Synset('savings_bank.n.02')

    [1] Lesk, Michael. "Automatic sense disambiguation using machine
    readable dictionaries: how to tell a pine cone from an ice cream
    cone." Proceedings of the 5th Annual International Conference on
    Systems Documentation. ACM, 1986.
    http://dl.acm.org/citation.cfm?id=318728
    """

    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    _, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return sense

for ss in wordnet.synsets('tolerance'):
    print(ss, ss.definition())