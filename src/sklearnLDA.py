from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
import pickle
from absl import flags
from absl import app
from wordcloud import WordCloud
import matplotlib.pyplot as plt

flags.DEFINE_integer(
    "num_topics",
    default=3,
    help="The number of topics.")
flags.DEFINE_integer(
    "num_top_words",
    default=10,
    help="The number of most likely words to display in a topic.")
flags.DEFINE_string(
    "master_data_dir",
    default="/Users/linyingzhang/git/zhangly811/WordSenseDetection/dat_unsync/output/",
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "word",
    default="ventricle",  # "/tmp/lda/data"
    help="Target word to disambiguate.")
flags.DEFINE_string(
    "source",
    default="Ped",  # "/tmp/lda/data"
    help="The data source.")
flags.DEFINE_string(
    "model_dir",
    default="/Users/linyingzhang/git/zhangly811/WordSenseDetection/res",  # "/tmp/lda/",
    help="Directory to put the model's fit.")

FLAGS = flags.FLAGS
FILE_TEMPLATE = "{split}.npy"

def load_dataset(directory, split_name):
    data = np.load(os.path.join(directory, FILE_TEMPLATE.format(split=split_name)), allow_pickle=True)
    data = np.array([list(word) for word in data], dtype=np.float32)
    return data

def main(argv):
    del argv  # unused

    data_dir = os.path.join(FLAGS.master_data_dir, "{}_{}".format(FLAGS.word, FLAGS.source))

    with open(os.path.join(data_dir, "dict.pkl"), "rb") as f:
        words_to_idx = pickle.load(f)
    num_words = len(words_to_idx)

    vocabulary = [None] * num_words
    for word, idx in words_to_idx.items():
        vocabulary[idx] = word

    dataset = load_dataset(data_dir, "train")
    if FLAGS.source == "WSH":
        dataset_test = load_dataset(data_dir, "test")
        dataset = np.concatenate((dataset, dataset_test), axis=0)

    lda = LatentDirichletAllocation(n_components=FLAGS.num_topics, random_state=7)
    lda.fit(dataset)

    # get topics for some given samples:
    topic_prob_per_doc = lda.transform(dataset)
    topic_assignment_per_doc = topic_prob_per_doc.argmax(axis=1)

    for topic in range(FLAGS.num_topics):
        freq = {}
        for idx in range(1, len(vocabulary)):
            if vocabulary[idx] != FLAGS.word:
                freq[vocabulary[idx]] = lda.components_[topic, idx]

        wordcloud = WordCloud(width=900, height=500, max_words=dataset.shape[1], relative_scaling=1,
                              normalize_plurals=False).generate_from_frequencies(freq)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    top_words_idx = np.array([topic[::-1] for topic in lda.components_.argsort(axis=1)])[:,:FLAGS.num_top_words]
    top_words_in_topics = []
    for topic in range(FLAGS.num_topics):
        top_words_in_topics.append(np.array([vocabulary[idx] for idx in top_words_idx[topic,:]]).reshape(1,-1))
    top_words_in_topics = np.array(top_words_in_topics).reshape(FLAGS.num_topics, FLAGS.num_top_words)
    for topic in range(FLAGS.num_topics):
        print("Topic {}: {}".format(topic, top_words_in_topics[topic,:]))


    # evaluation
    if FLAGS.source == "MSH" and FLAGS.word == "ventricle":
        trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)

        labels = np.concatenate((trainlabels, testlabels))

        labels = [label.replace('M1', '1') for label in labels] #M1 is heart
        labels = [label.replace('M2', '0') for label in labels] #M2 is brain
        labels = np.array(labels).astype(int)

        print ("Accuracy: ", np.mean(topic_assignment_per_doc == labels))

    if FLAGS.source == "MSH" and FLAGS.word == "cold":
        trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)

        labels = np.concatenate((trainlabels, testlabels))

        labels = [label.replace('M1', '0') for label in labels] #M1 is low temp
        labels = [label.replace('M2', '1') for label in labels] #M2 is COLD
        labels = [label.replace('M3', '2') for label in labels] #M3 is common cold

        labels = np.array(labels).astype(int)

        print ("Accuracy: ", np.mean(topic_assignment_per_doc == labels))



if __name__ == "__main__":
    app.run(main)

