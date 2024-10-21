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

def expand_query(query, synonym_dict):
    words = query.lower().split()  # Split the query into words
    expanded_words = []

    for word in words:
        # Replace the word with its synonym if it exists in the dictionary
        if word in synonym_dict:
            expanded_words.append(synonym_dict[word])
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)  # Return the expanded query

# Searching for documents
def search(query_string, synonym_dict):
    expanded_query = expand_query(query_string, synonym_dict)
    print(f"Original query: {query_string}")
    print(f"Expanded query: {expanded_query}")

    try:
        ix = open_dir(index_dir)
    except Exception as e:
        print(f"Error opening index: {e}")
        return ""

    # Parse the query string (search in name and content fields)
    query_parser = MultifieldParser(["name^2", "content"], ix.schema)
    query = query_parser.parse(expanded_query)

    # Search the index
    retrieved_docs = []
    with ix.searcher() as searcher:
        results = searcher.search(query)
        for result in results:
            #highlights = result.highlights("content")
            retrieved_docs.append(result['name'] + ": " + result['content'])  # Append the name and content
    
    # Return the combined results or an empty string if nothing is found
    return "\n".join(retrieved_docs) if retrieved_docs else "No results found for the query."

#queries = [
#    "Claudio Bravo",
#    "goalkeeper Chile",
#    "market value 2023",
#    "tallest player",
#    "highest market value",
#]
#
#for query in queries:
#    output = search(query)
#    print(f"Results for '{query}':\n{output}\n")

# Example user query
user_query = "messi valuation"

synonym_dict = {
    "goalie": "goalkeeper",
    "keeper": "goalkeeper",
    "valuation": "market value",
    "price": "market value",
    "contract end": "contract expiration",
    "worth": "market value",
    "tallest": "highest",
    "most expensive": "highest market value",
    "expensive": "market value",
}

output = search(user_query, synonym_dict)
print(output)