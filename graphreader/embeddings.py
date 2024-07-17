from sentence_transformers import SentenceTransformer


def get_embedding_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model


def embedding_function(text):
    model = get_embedding_model()
    embeddings = model.encode(text)
    return embeddings
