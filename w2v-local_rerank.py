# w2v-local rerank.sh [query-file] [top-100-file] [collection-file] [output-file] [expansions-file]
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
import time
from models import LanguageModel
import sys, os
import pickle
from params import STOPWORD_REMOVAL, LOWERCASE, STEMMING, MU, TOPK
query_file, top_100_file, collection_file, output_file, expansion_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
print(sys.argv)


# from query_id -> see top_100_file -> get doc_id -> 

# top_100_file is a tsv separated file with query_id, doc_id, score
# the first line is the header


set_of_docs = set()
def get_docs_from_query(top_100_file):
	docs_for_query = defaultdict(list)
	with open(top_100_file, 'r') as file:
		next(file)  # skip the header line
		for line in file:
			query_id, doc_id, score = line.strip().split('\t')
			docs_for_query[query_id].append(doc_id)
			set_of_docs.add(doc_id)
	return docs_for_query
def get_text_from_qid(query_file):
	text_from_qid = {}
	with open(query_file, 'r') as file:
		next(file)
		for line in file:
			qid, text = line.strip().split('\t')
			text_from_qid[qid] = text
	return text_from_qid

def get_doc_from_docid(docfile):
    # docid, url, title, body
	title = {}
	body = {}
	with open(docfile, 'r') as file:
		for line in file:
			x = ((line.strip().split('\t')))
			# docid, url, tit, bod = line.strip().split('\t')
			docid = x[0]
			if((docid in set_of_docs)):
				bod = x[-1]
				if(len(x) > 2):
					tit = x[2]
				else:
					tit = ''
				title[docid] =  tit
				body[docid] = bod
	return title, body
# for each query, get the languagee models

# with open('titles.pkl', 'wb') as
if os.path.exists('./titles.pkl'):
	print(f"exists")
	set_of_docs = pickle.load(open('set.pkl', 'rb'))
	titles = pickle.load(open('titles.pkl', 'rb'))
	bodies = pickle.load(open('bodies.pkl', 'rb'))
	text_from_qid = pickle.load(open('text_from_qid.pkl', 'rb'))
	docs_from_query = pickle.load(open('docs_from_query.pkl', 'rb'))
else:
	print(f"doesnot ")
	docs_from_query= get_docs_from_query(top_100_file)
	text_from_qid = get_text_from_qid(query_file)
	titles, bodies = get_doc_from_docid(collection_file)
	pickle.dump(set_of_docs, open('set.pkl', 'wb'))
	pickle.dump(titles, open('titles.pkl', 'wb'))
	pickle.dump(bodies, open('bodies.pkl', 'wb'))
	pickle.dump(text_from_qid, open('text_from_qid.pkl', 'wb'))
	pickle.dump(docs_from_query, open('docs_from_query.pkl', 'wb'))
    
print(len(set_of_docs))
# for qid in text_from_qid.keys():
# 	docs = docs_from_query[qid]
# 	for doc in docs:
# 		if(not (doc in set_of_docs)):
# 			print(f" doc = {doc}")
# 			break

print(f"starring loop")
write_file = output_file
with open(write_file, 'w') as file:
	pass
print(f"starting w2vec training")
t1 = time.time()
trained_model = {}
qlms = {}
doc_lmss = {}
corpus_lms = {}
for qid in text_from_qid.keys():
	# for each token in the query, we get a vector
	# mapping from token to numpy array
	# models[qid] -> token -> numpy array
	# U = V x d
	# U^t = d x V
	# query = V x 1
	# need word_counts for each word
	t2 = time.time()
	corpus_lm = LanguageModel([])
	docids = docs_from_query[qid]
	doc_lms = [LanguageModel([titles[docid], bodies[docid]]) for docid in docids]
	for i, doc_lm in enumerate(doc_lms):
		corpus_lm.combine_model(doc_lm)

	corpus_lm.referesh_probs()
	doc_lmss[qid] = doc_lms
	corpus_lms[qid] = corpus_lm
	qlm = LanguageModel([text_from_qid[qid]])
	qlm.dirichlet_smooth(corpus_lm)
	if(not os.path.exists(f'./w2v/{qid}.pkl')):
		if(not corpus_lm.tokens): print(f"corpus_lm.tokens is empty")
		# print((corpus_lm.tokens[0]))
		# sys.exit(-1)	
		model = Word2Vec(corpus_lm.tokens + qlm.tokens, vector_size=100, window=5,sg = 0, min_count=1, workers =os.cpu_count(), epochs=50)
		# Your existing model saving logic
		output_dir = './w2v/'
		# Create the directory if it doesn't exist
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		# Save the model
		pickle.dump(model, open(os.path.join(output_dir, f'{qid}.pkl'), 'wb'))
		# pickle.dump(model, open(f'./w2v/{qid}.pkl', 'wb'))
	else:
		model = pickle.load(open(f'./w2v/{qid}.pkl', 'rb'))
	# model = Word2Vec(corpus_lm.tokens, vector_size=100, window=5,sg = 0, min_count=1, workers =os.cpu_count())
	trained_model[qid] = model
	qlms[qid] = qlm
	print(f"qid = {qid} time taken = {time.time() - t2}")


	pass
print(f"ending w2vec training time = {time.time() - t1}")

with open(expansion_file, 'w') as file:
	pass
with open(write_file, 'w') as file:
	pass
for qid in text_from_qid.keys():
	t = time.time()
	model = trained_model[qid]
	V, k = model.wv.vectors.shape
	query_v = np.zeros((V, 1))
	### computing U U^t q
	qlm = qlms[qid]
	for word in qlm.word_counts.keys():
		if word in model.wv.key_to_index:
			query_v[model.wv.key_to_index[word], 0] = qlm.word_counts[word]
	U = model.wv.vectors
	Ut = U.T
	query_embeds = np.matmul(Ut, query_v)
	worddots = np.matmul(U, query_embeds)
	# d x V V x 1 
	# dim of worddots = V x 1
	top_indices = np.argsort(worddots.flatten())[-TOPK:]
	with open(expansion_file, 'a') as file:
		file.write(f"{qid} : ")
		pass
	new_toks = []
	for idx in top_indices:
		new_toks.append(model.wv.index_to_key[idx])
	with open(expansion_file, 'a') as file:
		file.write(' '.join(new_toks))
		file.write('\n')

	# we have the top -k tokens, we need to expand the query
	doc_lms = doc_lmss[qid]
	docs = docs_from_query[qid]
	corpus_lm = corpus_lms[qid]
	new_qlm = LanguageModel([text_from_qid[qid]])
	new_qlm.add_tokens(new_toks)
	new_qlm.dirichlet_smooth(corpus_lm)
	langkl = []
	# for (doc_lm, docid) in langmods:
	# 	langkl.append(( doc_lm.KL_div(qlm), docid ))
	for i, doc_lm in enumerate(doc_lms):
		docid = docs[i]
		langkl.append((doc_lm.rev_KL_div(new_qlm), docid))
	langkl.sort()
	# langmods.sort(key= lambda x : -x[0].KL_div(qlm))
	print(f"langkl = {langkl}")
	print(f"time taken = {time.time() - t}")
	with open(write_file, 'a') as file:
		for i, (kl, docid) in enumerate(langkl):
			file.write(f"{qid} Q0 {docid} {i + 1} {-kl} runid1\n")
# open(write_file, 'a').write('\n')
    
