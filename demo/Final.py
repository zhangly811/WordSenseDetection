import sys
import argparse
sys.path.insert(1, '../src')
import format_dat as FMT
import sklearnLDA_jw as LDA
import numpy as np
import os
from os import walk




def get_N_topics(word, f):
	fin = open(f, 'rt')
	#@RELATION C1321571_C0056208
	CUIs = fin.readline().split()[-1].split("_")
	fin.close()
	return CUIs

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d','--data', type=str, help = '')
	#parser.add_argument('-w','--word', type=str, help = '')
	#parser.add_argument('-f','--file', type=str, help = '')
	#parser.add_argument('--n_topcs', type=str, help = '')
	args = parser.parse_args()
	return args

def main():
	args = GetOptions()
	word, InpFil, n_topics = args.data.split()
	WORD = word
	n_topics = int(n_topics)
	word = word.lower()
	InpFil = "../MSHCorpus/" + InpFil
	source = "MSH"
	accs = []
	Nearwords = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 65, 70]
	for n_of_nearby_word in Nearwords:
		ins = FMT.DoBagOfWords(word.lower(), source, n_topics, InpFil, n_of_nearby_word)
		wc, acc, n_label = LDA.run(word.lower(), n_topics, source, num_top_words=n_of_nearby_word)
		accs.append(acc)
	acc = max(accs)
	bestK = Nearwords[accs.index(acc)]
	OutFil = open("res/{}.acc.txt".format(WORD), 'wt')
	OutFil.write(WORD+"\t"+str(acc)+"\t"+str(bestK)+"\t"+str(n_label)+"\n")
	OutFil.close()
	
main()
