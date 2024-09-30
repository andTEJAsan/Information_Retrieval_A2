from collections import defaultdict
from models import LanguageModel
import sys, os
query_file, top_100_file, collection_file, output_file, expansion_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
print(sys.argv)


# from query_id -> see top_100_file -> get doc_id -> 

# top_100_file is a tsv separated file with query_id, doc_id, score
# the first line is the header


def get_docs_from_query(top_100_file):
	docs_for_query = defaultdict(list)
	with open(top_100_file, 'r') as file:
		next(file)  # skip the header line
		for line in file:
			query_id, doc_id, score = line.strip().split('\t')
			docs_for_query[query_id].append(doc_id)

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
			bod = x[-1]
			if(len(x) > 2):
				tit = x[2]
			else:
				tit = ''
			title[docid] =  tit
			body[docid] = bod
	return title, body
# for each query, get the languagee models

titles, bodies = get_doc_from_docid(collection_file)
text_from_qid = get_text_from_qid(query_file)
docs_from_query= get_docs_from_query(top_100_file)

for qid in text_from_qid.keys():
	corpus_lm = LanguageModel()
	docids = docs_from_query[qid]
	langmods = []
	for docid in docids:
		doc_lm = LanguageModel([titles[docid], bodies[docid]])
		corpus_lm.combine_model(doc_lm)
		langmods.append((doc_lm, docid))
	qlm = LanguageModel([text_from_qid[qid]])


    
