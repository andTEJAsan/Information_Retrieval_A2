from collections import defaultdict
import sys, os
query_file, top_100_file, collection_file, output_file, expansion_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]


docs_for_query = defaultdict(list)	
# from query_id -> see top_100_file -> get doc_id -> 

# top_100_file is a tsv separated file with query_id, doc_id, score
# the first line is the header

with open(top_100_file, 'r') as file:
	next(file)  # skip the header line
	for line in file:
		query_id, doc_id, score = line.strip().split('\t')
		docs_for_query[query_id].append(doc_id)