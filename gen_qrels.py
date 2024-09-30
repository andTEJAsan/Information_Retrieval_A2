input_file = './qrels.tsv'
output_file = './gen_qrels.txt'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
	next(f_in)
	for line in f_in:
		query_id, doc_id, relevance, iteration = line.strip().split('\t')
		f_out.write(f'{query_id} 0 {doc_id} {relevance}\n')