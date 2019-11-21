import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import argparse
import sys
import csv
import random
import numpy as np
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))
wnl = WordNetLemmatizer()


def parseArg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--InpFil', required=True, type=str,
                        help='Txt file contains search results from Pubmed')
    parser.add_argument('-o', '--outname', default="train.npy", type=str,
                        help='Txt file contains search results from Pubmed')
    args = parser.parse_args()
    return args


class GetPubmed(object):
    def __init__(self, args, trainP=0.7):
        self.InpFil = args.InpFil
        self.outname = args.outname
        self.trainFil = "%s_train.npy" % self.outname
        self.testFil = "%s_test.npy" % self.outname
        self.trainLabel = "%s_train.labels" % self.outname
        self.testLabel = "%s_test.labels" % self.outname
        self.DictFil = "%s_dict.pkl" % self.outname
        self.pkl = {}
        self.trainP = trainP
        self.key_level = 0
        self.word2key = {"cold": 1, " ": 0}
        self.key2word = {1: "cold", 0: " "}
        self.TrainArray = []
        self.TestArray = []
        self.TrainLabels = []
        self.TestLabels = []

    def run(self):
        InpFil = open(self.InpFil, "rt")
        SaveFil = open(self.DictFil, "wt")
        Counter = 0
        reader = csv.reader(InpFil)
        for row in reader:
            if len(row) < 3 or row[0].startswith("@"):
                # print(row)
                continue
            self.processAbstract(row[1].lower(), row[2])

            # break
        # print(Counter)
        output = open(self.DictFil, 'wb')
        pickle.dump(self.word2key, output)
        output.close()

        tmp = np.empty(len(self.TrainArray), object)
        tmp[:] = self.TrainArray
        np.save(self.trainFil, tmp)
        tmp = np.empty(len(self.TestArray), object)
        tmp[:] = self.TestArray
        np.save(self.testFil, tmp)
        #np.save(self.trainFil, np.array(self.TrainArray, dtype='object'))
        #np.save(self.testFil, np.array(self.TestArray, dtype='object'))

        output = open(self.trainLabel, 'wt')
        output.write("\n".join(self.TrainLabels))
        output.close()
        output = open(self.testLabel, 'wt')
        output.write("\n".join(self.TestLabels))
        output.close()

    def processAbstract(self, abstract, label):
        abstract = abstract.replace("-", " ")
        abstract = abstract.translate(
            str.maketrans('', '', string.punctuation))
        abstract = abstract.replace("colds", "cold")
        #abstract = abstract.split()
        # print(abstract)
        #abstract = re.split('-;, ',abstract)
        # print(abstract)

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(abstract)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        # print(word_tokens)
        # print(filtered_sentence)
        abstract = filtered_sentence

        try:
                #idx = abstract.index("<e>cold</e>")
            idx = abstract.index("ecolde")
        except:
            print("Error", abstract)
            # exit()
            return
        # print(idx)
        words = abstract  # .split()
        word_array = []
        test_words = []
        for i in range(idx - 10, idx + 10):
            if i < 0 or i > len(words) - 1:
                word_array.append(0)
                continue
            #print(i, len(words))
            try:
                word = words[i]
            except:
                print(words, i)
            if word == "ecolde":
                word = 'cold'
            elif word == "ecoldse":
                word = 'colds'
            else:
                word = word.translate(
                    str.maketrans('', '', string.punctuation))
            if word not in self.word2key:
                self.key_level += 1
                self.word2key[word] = self.key_level
                self.key2word[self.key_level] = word
            key = self.word2key[word]
            word_array.append(key)
            test_words.append(word)
        print(test_words)
        word_array = np.array(word_array, dtype='float64')
        if random.random() < self.trainP:
            #outfil = self.testFil
            self.TrainArray.append(word_array)
            self.TrainLabels.append(label)
        else:
            #outfil = self.trainFil
            self.TestArray.append(word_array)
            self.TestLabels.append(label)


class MyBagOfWords(object):
    def __init__(self, args, trainP=0.7):
        self.the_word = wnl.lemmatize(args.word)
        self.source = args.source
        self.InpFil = args.InpFil
        self.outname = args.outname
        self.trainFil = "%s/train.npy" % self.outname
        self.testFil = "%s/test.npy" % self.outname
        if self.source == "MSH":
            self.trainLabel = "%s/train.labels" % self.outname
            self.testLabel = "%s/test.labels" % self.outname
        self.DictFil = "%s/dict.pkl" % self.outname
        self.pkl = {}
        self.trainP = trainP
        self.key_level = 0
        #self.word2key = {self.the_word:1, " ":0}
        #self.key2word = {1:self.the_word, 0:" "}
        self.word2key = {}  # {self.the_word:0}
        self.key2word = {}  # {0:self.the_word}
        self.TrainArray = []
        self.TestArray = []
        self.TrainLabels = []
        self.TestLabels = []
        self.n_of_nearby_word = args.n_of_nearby_word
        self.Data = []
        self.Labels = []
        self.PubMedIDs = []
        self.WC = {}

    def run(self):
        if self.source == "MSH":
            if not os.path.exists(self.outname):
                os.mkdir(self.outname)
            InpFil = open(self.InpFil, "rt")
            SaveFil = open(self.DictFil, "wt")
            Counter = 0
            reader = csv.reader(InpFil)
            for row in reader:
                if len(row) < 3 or row[0].startswith("@"):
                    continue
                #self.processAbstract(row[1].lower(), row[2])
                self.corpusWC(row[1].lower())
                self.Labels.append(row[2])
                self.PubMedIDs.append(row[0])
            self.KeptWords = [k for k, v in self.WC.items() if v >= 10]
            for i, word in enumerate(self.KeptWords):
                if word not in self.word2key:
                    self.word2key[word] = i
                    self.key2word[i] = word
            self.KeptWords = set(self.KeptWords)
            for pubmedID, lemmatized_words, label in zip(self.PubMedIDs, self.Data, self.Labels):
                #print(lemmatized_words)
                self.processAbstract(pubmedID, lemmatized_words, label)
            output = open(self.DictFil, 'wb')
            pickle.dump(self.word2key, output)
            output.close()

            tmp = np.empty(len(self.TrainArray), object)
            tmp[:] = self.TrainArray
            np.save(self.trainFil, tmp)
            tmp = np.empty(len(self.TestArray), object)
            tmp[:] = self.TestArray
            np.save(self.testFil, tmp)

            output = open(self.trainLabel, 'wt')
            output.write("\n".join(self.TrainLabels))
            output.close()
            output = open(self.testLabel, 'wt')
            output.write("\n".join(self.TestLabels))
            output.close()

        elif self.source == "Ped":
            if not os.path.exists(self.outname):
                os.mkdir(self.outname)
            InpFil = open(self.InpFil, "rt")
            SaveFil = open(self.DictFil, "wt")
            AbsFil = open("%s/Abstracts.txt" % self.outname, "wt")
            Counter = 0
            for l in InpFil:
                try:
                    abstract = l.split("|")[2]
                    # AbsFil.write(abstract+"\n")
                    self.corpusWC2(abstract.lower(), AbsFil)
                except:
                    # print(abstract)
                    continue
            # print(self.WC)
            # return
            #self.KeptWords = set([k for k,v in self.WC.items() if v >= 10])
            self.KeptWords = [k for k, v in self.WC.items() if v >= 10]
            for i, word in enumerate(self.KeptWords):
                if word not in self.word2key:
                    self.word2key[word] = i
                    self.key2word[i] = word
            self.KeptWords = set(self.KeptWords)
            for lemmatized_words in zip(self.Data):
                # print(lemmatized_words)
                self.processAbstract2(lemmatized_words)

            output = open(self.DictFil, 'wb')
            pickle.dump(self.word2key, output)
            output.close()

            tmp = np.empty(len(self.TrainArray), object)
            tmp[:] = self.TrainArray
            np.save(self.trainFil, tmp)
            tmp = np.empty(len(self.TestArray), object)
            tmp[:] = self.TestArray
            np.save(self.testFil, tmp)

    def preprocessWords(self, abstract):
        abstract = abstract.replace("-", " ")
        abstract = abstract.replace("'", " ")
        abstract = abstract.replace(".", " ")
        #abstract = re.sub("was", "is", abstract)
        abstract = re.sub("<e>", " ", abstract)
        abstract = re.sub("</e>", " ", abstract)
        #abstract = abstract.replace("/", " ")
        abstract = abstract.translate(str.maketrans(
            ' ', ' ', string.punctuation))  # remove punctuations
        # abstract = abstract.translate(str.maketrans(string.punctuation, ' ')) # remove punctuations
        tokens = [token.lower() for token in word_tokenize(
            abstract)]  # lowercase and tokenize
        lemmatized_words = [wnl.lemmatize(token)
                            for token in tokens]  # limmatize
        # remove stop words
        lemmatized_words = [w for w in lemmatized_words if not w in stop_words]
        # self.Data.append(lemmatized_words)
        return lemmatized_words

    def corpusWC(self, abstract):
        lemmatized_words = self.preprocessWords(abstract)
        already = set([])
        self.Data.append(lemmatized_words)
        for word in lemmatized_words:
            if word not in self.WC:
                self.WC[word] = 0
            if word not in already:
                self.WC[word] += 1
                already.add(word)

    def processAbstract(self, pubmed, lemmatized_words, label):
        lemmatized_words = [
            word for word in lemmatized_words if word in self.KeptWords]
        try:
            idx = lemmatized_words.index(self.the_word)
        except:
            print("Error", pubmed, lemmatized_words)
            return
        #word_array = []
        """
		for i in range(idx-10,idx+10):
			if i < 0 or i > len(lemmatized_words)-1:
				word_array.append(0)
				continue
			try:
				word = lemmatized_words[i]
			except:
				print(lemmatized_words, i)
			if word not in self.word2key:
				self.key_level += 1
				self.word2key[word] = self.key_level
				self.key2word[self.key_level] = word
			key = self.word2key[word]
			word_array.append(key)
		"""
        """
		for i in range(0, len(lemmatized_words)):
			word = lemmatized_words[i]
			if word not in self.word2key:
				self.key_level += 1
				self.word2key[word] = self.key_level
				self.key2word[self.key_level] = word
			key = self.word2key[word]
			word_array.append(key)
		word_array = np.array(word_array, dtype='float64')
		"""
        word_window = lemmatized_words[max(
            0, idx - 10): min(idx + 10, len(lemmatized_words))]
        word_array = np.zeros(len(self.KeptWords))
        for i in range(0, len(word_window)):
            word = word_window[i]
            num = word_window.count(word)
            idx = self.word2key[word]
            #word_array[idx] = self.WC[word]
            word_array[idx] = num

        if random.random() < self.trainP:
            self.TrainArray.append(word_array)
            self.TrainLabels.append(label)
        else:
            self.TestArray.append(word_array)
            self.TestLabels.append(label)

    def countNumofWord(self, word, method=1):
        if method == 1:
            wc = 0
            for dat in self.Data:
                dat = [word for word in dat if word in self.KeptWords]
                wc += dat.count(word)
            return wc, self.WC[word]
        else:
            return self.WC[word]

    def corpusWC2(self, abstract, AbsFil):
        already = set([])
        lemmatized_words = self.preprocessWords(abstract)
        #lemmatized_words = lemmatized_words[0]
        # print(lemmatized_words)
        try:
            idx = lemmatized_words.index(self.the_word)
            AbsFil.write(abstract + "\n")
            self.Data.append(lemmatized_words)
            for word in lemmatized_words:
                if word not in self.WC:
                    self.WC[word] = 0
                if word not in already:
                    self.WC[word] += 1
                    already.add(word)
        except:
            #print("Error", lemmatized_words)
            return

    def processAbstract2(self, lemmatized_words):
        lemmatized_words = lemmatized_words = [
            word for word in lemmatized_words[0] if word in self.KeptWords]
        # print(lemmatized_words)
        word_array = []
        try:
            idx = lemmatized_words.index(self.the_word)
        except:
            print("Error", self.the_word, lemmatized_words)
            return

        word_window = lemmatized_words[max(
            0, idx - 10): min(idx + 10, len(lemmatized_words))]
        word_array = np.zeros(len(self.KeptWords))
        for i in range(0, len(word_window)):
            word = word_window[i]
            num = word_window.count(word)
            idx = self.word2key[word]
            #word_array[idx] = self.WC[word]
            word_array[idx] = num
        word_array = np.array(word_array, dtype='float64')
        self.TrainArray.append(word_array)

    # def generatewordvector(self)

class arguments:
    def __init__(self, word, InpFil, outname, trainP, n_of_nearby_word, source):
        self.InpFil = InpFil
        self.outname = outname
        self.trainP = trainP
        self.word = word
        self.n_of_nearby_word = n_of_nearby_word
        self.source = source

def DoBagOfWords(word, source, n_of_nearby_word = 10):
    if source == "MSH":
        args = arguments(word, "../MSHCorpus/%s_pmids_tagged.arff"%(word), "../output/%s_MSH"%word, 0.7, 10, "MSH")
        ins = MyBagOfWords(args)
        ins.run()
    elif source == "Pediatrics":
        args = arguments(word, "../dat/Pediatrics.Abs.txt", "../output/%s_Ped"%word, None, 10, "Ped")
        ins = MyBagOfWords(args)
        ins.run()
    return ins

def main():
    args = parseArg()
    ins = GetPubmed(args)
    ins.run()


if __name__ == '__main__':
    main()
