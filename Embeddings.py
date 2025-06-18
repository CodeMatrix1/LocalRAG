from langchain.embeddings import HuggingFaceEmbeddings
def embeddings_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")