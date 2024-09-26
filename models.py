import math
import nltk
import typing
import re
from params import STOPWORD_REMOVAL, LOWERCASE, STEMMING

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
class LanguageModel:
	def __init__(self, sentences):
		self.sentences = sentences
		self.tokens = self.tokenize_sentences()
		self.probs = dict()
		self.word_counts = self.count_occurrences()
		self.base_model = None
		self.mu = None

	def tokenize_sentences(self):
		tot_tokens = []
		for sentence in self.sentences:
			tokens = nltk.word_tokenize(sentence)
			if STOPWORD_REMOVAL:
				tokens = [token for token in tokens if token.lower() not in nltk.corpus.stopwords.words('english')]
			if LOWERCASE:
				tokens = [token.lower() for token in tokens]
			if STEMMING:
				stemmer = nltk.stem.PorterStemmer()
			tokens = [stemmer.stem(token) for token in tokens]
			tot_tokens.extend(tokens)
		return tot_tokens

	def count_occurrences(self):
		word_counts = {}
		for token in self.tokens:
			if token in word_counts:
				word_counts[token] += 1
			else:
				word_counts[token] = 1
		for word in word_counts:
			self.probs[word] = word_counts[word] / len(self.tokens)
		return word_counts
	def dirichlet_smooth(self, mu, collection_model):
		for word in self.word_counts:
			self.probs[word] = (self.word_counts[word] + mu * collection_model.probs[word]) / (len(self.tokens) + mu)
	def probability(self, word):
		return self.probs.get(word, 0)
	# computes D(M_s || M_c) = sum_{w in V} P(w|M_s) log (P(w|M_s) / P(w|M_c))
	def KL_divergence(self, model):
		KL = 0 
		for word in self.probs:
			KL += self.probs[word] * math.log(self.probs[word] / model.pro)


# Example usage
