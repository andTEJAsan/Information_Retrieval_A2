
# Check for duplicates in outputfile.txt
with open('outputfile.txt', 'r') as f:
    lines = f.readlines()
    seen = set()
    for line in lines:
        parts = line.split()
        
        # Skip lines that don't have the expected number of columns
        if len(parts) < 6:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        
        query_id, doc_id = parts[0], parts[2]  # Query ID and Doc ID
        if (query_id, doc_id) in seen:
            print(f"Duplicate found: Query ID = {query_id}, Doc ID = {doc_id}")
        else:
            seen.add((query_id, doc_id))
# Check for duplicates in outputfile.txt
with open('outputfile.txt', 'r') as f:
    lines = f.readlines()
    seen = set()
    for line in lines:
        parts = line.split()
        query_id, doc_id = parts[0], parts[2]  # Query ID and Doc ID
        if (query_id, doc_id) in seen:
            print(f"Duplicate found: Query ID = {query_id}, Doc ID = {doc_id}")
        else:
            seen.add((query_id, doc_id))

# Check for duplicates in qrels.tsv
with open('qrels.tsv', 'r') as f:
    lines = f.readlines()
    seen = set()
    for line in lines:
        parts = line.split()
        query_id, doc_id = parts[0], parts[1]
        if (query_id, doc_id) in seen:
            print(f"Duplicate found: {query_id}, {doc_id}")
        else:
            seen.add((query_id, doc_id))