from collections import defaultdict
import time
from models import LanguageModel
import sys, os
import pickle
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

for qid in text_from_qid.keys():

	t = time.time()
	corpus_lm = LanguageModel([])
	docids = docs_from_query[qid]
	langmods = []
	# for docid in docids:
	# 	try:
	# 		assert(docid in titles)
	# 	except:
	# 		print(f"docid = {docid}")
	# 	assert(docid in bodies)

	doc_lms = [LanguageModel([titles[docid], bodies[docid]]) for docid in docids]
	print(f"time.time - t = {time.time() - t}")

	t = time.time()
	for i, doc_lm in enumerate(doc_lms):
		corpus_lm.combine_model(doc_lm)
		langmods.append((doc_lm, docids[i]))
	corpus_lm.referesh_probs()
	qlm = LanguageModel([text_from_qid[qid]])
	qlm.dirichlet_smooth(corpus_lm)
	langkl = []
	for (doc_lm, docid) in langmods:
		langkl.append(( -doc_lm.KL_div(qlm), docid ))
	langkl.sort()
	# langmods.sort(key= lambda x : -x[0].KL_div(qlm))
	print(f"langkl = {langkl}")
	print(f"time taken = {time.time() - t}")
	with open(write_file, 'a') as file:
		for i, (kl, docid) in enumerate(langkl):
			file.write(f"{qid} Q0 {docid} {i + 1} {-kl} runid1\n")

open(write_file, 'a').write('\n')
    
