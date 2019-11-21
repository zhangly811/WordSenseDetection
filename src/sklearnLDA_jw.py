from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt


FILE_TEMPLATE = "{split}.npy"

def load_dataset(directory, split_name):
    data = np.load(os.path.join(directory, FILE_TEMPLATE.format(split=split_name)), allow_pickle=True)
    data = np.array([list(word) for word in data], dtype=np.float32)
    return data

def run(Word, num_topics, source, num_top_words=10, master_data_dir="../output/"):
    data_dir = os.path.join(master_data_dir, "{}_{}".format(Word, source))

    with open(os.path.join(data_dir, "dict.pkl"), "rb") as f:
        words_to_idx = pickle.load(f)
    num_words = len(words_to_idx)

    vocabulary = [None] * num_words
    for word, idx in words_to_idx.items():
        vocabulary[idx] = word

    dataset = load_dataset(data_dir, "train")
    if source == "WSH":
        dataset_test = load_dataset(data_dir, "test")
        dataset = np.concatenate((dataset, dataset_test), axis=0)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=7)
    lda.fit(dataset)

    # get topics for some given samples:
    topic_prob_per_doc = lda.transform(dataset)
    topic_assignment_per_doc = topic_prob_per_doc.argmax(axis=1)

    res = [] 
    for topic in range(num_topics):
        freq = {}
        for idx in range(1, len(vocabulary)):
            if vocabulary[idx] != word:
                freq[vocabulary[idx]] = lda.components_[topic, idx]

        wordcloud = WordCloud(width=1080, height=720, max_words=dataset.shape[1], relative_scaling=1,
                              normalize_plurals=False).generate_from_frequencies(freq)
        res.append(wordcloud)
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.show()

    top_words_idx = np.array([topic[::-1] for topic in lda.components_.argsort(axis=1)])[:,:num_top_words]
    top_words_in_topics = []
    for topic in range(num_topics):
        top_words_in_topics.append(np.array([vocabulary[idx] for idx in top_words_idx[topic,:]]).reshape(1,-1))
    top_words_in_topics = np.array(top_words_in_topics).reshape(num_topics, num_top_words)
    for topic in range(num_topics):
        print("Topic {}: {}".format(topic, top_words_in_topics[topic,:]))


    # evaluation
    if source == "MSH" and Word == "ventricle":
        trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)

        labels = np.concatenate((trainlabels, testlabels))

        labels = [label.replace('M1', '1') for label in labels] #M1 is heart
        labels = [label.replace('M2', '0') for label in labels] #M2 is brain
        labels = np.array(labels).astype(int)

        print ("Accuracy: ", np.mean(topic_assignment_per_doc == labels))
    
    if source == "MSH" and Word == "cold":
        trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)

        labels = np.concatenate((trainlabels, testlabels))

        labels = [label.replace('M1', '0') for label in labels] #M1 is low temp
        labels = [label.replace('M2', '1') for label in labels] #M2 is COLD
        labels = [label.replace('M3', '2') for label in labels] #M3 is common cold

        labels = np.array(labels).astype(int)

        print ("Accuracy: ", np.mean(topic_assignment_per_doc == labels))

    return res


if __name__ == "__main__":
    app.run(main)

