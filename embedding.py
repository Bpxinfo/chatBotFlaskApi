import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
#from azure_files_share import upload_EmbeddingFile
from azure_blob_storage import upload_vector_metadata
import pickle

def query_embedding(query):
    # Create a SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True).numpy()
    return query_embedding

def text_embedding(chunks, filename):
    # Create a SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    embeddings_np = np.array(chunk_embeddings)
    
    # Store metadata like chunk ID and original text
    metadata = {i: {'chunk': chunks[i], 'text': chunks[i]} for i in range(len(chunks))}
    
    # Initialize FAISS index for L2 distance
    d = embeddings_np.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)
    
    # Add the embeddings to the FAISS index
    index.add(embeddings_np)
    
    # Save the FAISS index directly to Azure File Share
    faiss_index_bytes = faiss.serialize_index(index)  # Serialize FAISS index into bytes
    #upload_EmbeddingFile(faiss_index_bytes, f"{filename}.txt","vectorsfiles")
    upload_vector_metadata(faiss_index_bytes,f"{filename}.index","vectorsfiles")

    # Save the metadata directly to Azure File Share
    metadata_bytes = pickle.dumps(metadata)  # Serialize metadata into bytes
    #upload_EmbeddingFile(metadata_bytes, f"{filename}.txt","metadafiles")
    upload_vector_metadata(metadata_bytes,f"{filename}.pkl","metadafiles")


