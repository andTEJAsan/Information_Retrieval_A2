
import sys
import numpy as np
import math
import nltk
import typing
import re
from params import STOPWORD_REMOVAL, LOWERCASE, STEMMING, MU, TOPK
DELIMITERS =[' ', ',', '.', ':', ';', '"', '\'', '@', '$', '%', '*', '/']
# nltk.download('popular')
def parse_tsv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        # Split the line by tab character
        fields = re.split(r'\t', line.strip())
        data.append(fields)
    return data

# Example usage
# docs.tsv contains doc_id, url, title, body as tab-separated values
# queries.tsv contains query_id, text as tab-separated values
# qrels.tsv contains query_id, doc_id , relevance, iteration as tab-separated values
# we are removing digits completely from the text

# Just keep working, you will reap the results later. It's a long journey, but it's worth it.
class LanguageModel:
	def __init__(self, sentences):
		self.sentences = sentences
		self.length = 0
		self.tokens = self.tokenize_sentences()
		self.probs = dict()
		self.word_counts = {}
		self.count_occurrences()
		self.base_model = None
		self.mu = MU
		self.smoother = None
	def add_tokens(self, toks):
		self.tokens.extend(toks)
		self.length += len(toks)
		for token in toks:
			if token in self.word_counts:
				self.word_counts[token] += 1
			else:
				self.word_counts[token] = 1
		self.referesh_probs()
	def tokenize_sentences(self):
		tot_tokens = []
		for sentence in self.sentences:
			sentence = re.sub(r'\d', '', sentence)
			sentence = re.sub(r'[{}]'.format(''.join(DELIMITERS)), ' ', sentence)
			tokens = nltk.word_tokenize(sentence)
			if STOPWORD_REMOVAL:
				tokens = [token for token in tokens if token.lower() not in nltk.corpus.stopwords.words('english')]
			if LOWERCASE:
				tokens = [token.lower() for token in tokens]
			if STEMMING:
				stemmer = nltk.stem.PorterStemmer()
				tokens = [stemmer.stem(token) for token in tokens]
			self.length += len(tokens)
			tot_tokens.append(tokens)
		return tot_tokens

	def count_occurrences(self):
		word_counts = self.word_counts
		for tokens in self.tokens:
			for token in tokens:
				if token in word_counts:
					word_counts[token] += 1
				else:
					word_counts[token] = 1
		for word in word_counts:
			self.probs[word] = word_counts[word] / self.length
	def dirichlet_smooth(self,collection_model):
		mu = self.mu
		self.smoother = collection_model	
		for word in self.word_counts:
			self.probs[word] = (self.word_counts[word] + mu * collection_model.probs.get(word, 0)) / (self.length + mu)
		for word in collection_model.word_counts:
			if word not in self.probs:
				self.probs[word] = mu * collection_model.probs[word] / (self.length + mu)
	def combine_model(self, model):
		self.tokens.extend(model.tokens)
		self.length += model.length
		for word in model.word_counts:
			if word in self.word_counts:
				self.word_counts[word] += model.word_counts[word]
			else:
				self.word_counts[word] = model.word_counts[word]
	def referesh_probs(self):
		for word in self.word_counts:
			self.probs[word] = self.word_counts[word] / self.length
	def probability(self, word):
		return self.probs.get(word, 0)
	# computes D(M_s || M_c) = sum_{w in V} P(w|M_s) log (P(w|M_s) / P(w|M_c))
	def probability(self, word, background_smoother):
		tf = self.word_counts.get(word, 0)
		return (tf + self.mu * background_smoother.probs.get(word, 0)) / (self.length + self.mu)

	def KL_div(self, rel_mod):
		# self = doc_model
		# rel_mod = query_model
		rel_probs = []
		doc_probs = []
		for word in rel_mod.probs.keys():
			rel_probs.append(rel_mod.probs[word])
			doc_probs.append(self.probability(word, rel_mod))
		rel_probs = np.array(rel_probs)
		doc_probs = np.array(doc_probs)
		kl_div = np.sum(doc_probs * np.log(doc_probs / rel_probs))
		return kl_div
	def rev_KL_div(self, rel_mod):
		# self = doc_model
		# rel_mod = query_model
		rel_probs = []
		doc_probs = []
		for word in rel_mod.probs.keys():
			rel_probs.append(rel_mod.probs[word])
			doc_probs.append(self.probability(word, rel_mod))
		rel_probs = np.array(rel_probs)
		doc_probs = np.array(doc_probs)
		kl_div = np.sum(rel_probs * np.log(rel_probs / doc_probs))
		return kl_div




# Example usage
