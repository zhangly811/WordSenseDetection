from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import itertools

FILE_TEMPLATE = "{split}.npy"

def load_dataset(directory, split_name):
    data = np.load(os.path.join(directory, FILE_TEMPLATE.format(split=split_name)), allow_pickle=True)
    data = np.array([list(word) for word in data], dtype=np.float32)
    return data

def run(Word, num_topics, source, num_top_words=10, master_data_dir="../output/", verbose=0):
    data_dir = os.path.join(master_data_dir, "{}_{}".format(Word, source))

    with open(os.path.join(data_dir, "dict.pkl"), "rb") as f:
        words_to_idx = pickle.load(f)
    num_words = len(words_to_idx)

    vocabulary = [None] * num_words
    for word, idx in words_to_idx.items():
        vocabulary[idx] = word

    dataset = load_dataset(data_dir, "train")
    if source == "MSH":
        dataset_test = load_dataset(data_dir, "test")
        dataset = np.concatenate((dataset, dataset_test), axis=0)
    #print(dataset.shape)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=7)
    lda.fit(dataset)

    # get topics for some given samples:
    topic_prob_per_doc = lda.transform(dataset)
    topic_assignment_per_doc = topic_prob_per_doc.argmax(axis=1)

    wordclouds = [] 
    for topic in range(num_topics):
        freq = {}
        for idx in range(1, len(vocabulary)):
            #if vocabulary[idx] != Word and vocabulary[idx] != "wa":
            if vocabulary[idx] != Word: #and vocabulary[idx] != "wa":
                freq[vocabulary[idx]] = lda.components_[topic, idx]
            #freq[vocabulary[idx]] = lda.components_[topic, idx]

        wordcloud = WordCloud(width=1080, height=720, max_words=dataset.shape[1], relative_scaling=1,
                              normalize_plurals=False).generate_from_frequencies(freq)
        wordclouds.append(wordcloud)
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.show()
    if source == "Ped":
        top_words_idx = np.array([topic[::-1] for topic in lda.components_.argsort(axis=1)])[:,:10]
    else: 
        top_words_idx = np.array([topic[::-1] for topic in lda.components_.argsort(axis=1)])[:,:num_top_words]
    top_words_in_topics = []
    for topic in range(num_topics):
        top_words_in_topics.append(np.array([vocabulary[idx] for idx in top_words_idx[topic,:]]).reshape(1,-1))
    top_words_in_topics = np.array(top_words_in_topics).reshape(num_topics, num_top_words)

    if verbose == 1 :
        for topic in range(num_topics):
            print("Topic {}: {}".format(topic, [x for x in top_words_in_topics[topic,:] if x != "wa"]))


    # evaluation
    if source == "MSH" and Word == "ventricle":
        trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)

        labels = np.concatenate((trainlabels, testlabels))

        labels = [label.replace('M1', '1') for label in labels] #M1 is heart
        labels = [label.replace('M2', '0') for label in labels] #M2 is brain
        labels = np.array(labels).astype(int)

        #print ("Accuracy: ", np.mean(topic_assignment_per_doc == labels))
    
    if source == "MSH" and Word == "cold":
        trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)

        labels = np.concatenate((trainlabels, testlabels))

        labels = [label.replace('M1', '0') for label in labels] #M1 is low temp
        labels = [label.replace('M2', '1') for label in labels] #M2 is COLD
        labels = [label.replace('M3', '2') for label in labels] #M3 is common cold

        labels = np.array(labels).astype(int)
 
    if source == "Ped" and Word == "ventricle":
        #trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        #testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)
        #labels = np.concatenate((trainlabels, testlabels))
        labels = np.loadtxt("../dat/ventricle.labels", dtype=np.str) 
        labels = [label.replace('M1', '1') for label in labels] #M1 is heart
        labels = [label.replace('M2', '0') for label in labels] #M1 is heart
        labels = np.array(labels).astype(int)
        #print(len(topic_assignment_per_doc), topic_assignment_per_doc)
        #print(len(labels), labels)
    
    if source == "MSH":
        trainlabels = np.loadtxt(os.path.join(data_dir, "train.labels"),dtype=np.str)
        testlabels = np.loadtxt(os.path.join(data_dir, "test.labels"), dtype=np.str)
        labels = np.concatenate((trainlabels, testlabels))
        
        highestACC = 0
        N_labels = len(set(labels))
        labels_set = [str(x) for x in range(N_labels)]
        for permute in itertools.permutations(labels_set):
            This_Label = labels.copy()
            for i, j in enumerate(permute):
                This_Label = [label.replace('M{}'.format(i+1), j) for label in This_Label]
            #print(labels)
            This_Label = np.array(This_Label).astype(int)
            acc = np.mean(topic_assignment_per_doc == This_Label)
            if acc > highestACC:
               highestACC = acc

    
    if verbose == 2:
        #print(len(topic_assignment_per_doc), topic_assignment_per_doc)
        #print(len(labels), labels)
        print ("Accuracy: ", np.mean(topic_assignment_per_doc == labels))

    return wordclouds, highestACC, len(labels)

if __name__ == "__main__":
    app.run(main)

