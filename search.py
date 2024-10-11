from indexing import index_dir
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
import os

# Global cache for the index
_cached_index = None

def get_index():
    """Load the Whoosh index once and cache it for further reuse."""
    global _cached_index
    if _cached_index is None:
        if os.path.exists(index_dir):
            _cached_index = open_dir(index_dir)
        else:
            raise FileNotFoundError(f"Index directory '{index_dir}' does not exist.")
    return _cached_index

# Searching for documents
def search(query_string):
    try:
        ix = open_dir(index_dir)
    except Exception as e:
        print(f"Error opening index: {e}")
        return ""

    # Parse the query string (search in name and content fields)
    query_parser = MultifieldParser(["name", "content"], ix.schema)
    query = query_parser.parse(query_string)

    # Search the index
    retrieved_docs = []
    with ix.searcher() as searcher:
        results = searcher.search(query)
        for result in results:
            #highlights = result.highlights("content")
            retrieved_docs.append(result['name'] + ": " + result['content'])  # Append the name and content
    
    # Return the combined results or an empty string if nothing is found
    return "\n".join(retrieved_docs) if retrieved_docs else "No results found for the query."

# Example user query
user_query = "goalkeeper chile"
output = search(user_query)
print(output)