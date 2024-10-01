from azure.storage.blob import BlobServiceClient
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Azure Storage Account details
connection_string = os.getenv("AZURE_BLOB_CONNECTIONSTRING")

def upload_vector_metadata(content,filename,container):

    if isinstance(content, np.ndarray):
        content = content.tobytes()  # Convert numpy array to bytes
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get the container client
    container_client = blob_service_client.get_container_client(container)

    # Get the blob client (for the specific file)
    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(content, overwrite=True)  # Set overwrite=True to replace existing blob
    
    # # Download the blob's content
    # with open("output.txt", "wb") as file:
    #     download_stream = blob_client.download_blob()
    #     file.write(download_stream.readall())
    #     # Read the blob's content into memory
    # blob_data = blob_client.download_blob().readall()

    # # Convert the content to string (assuming it's a text file)
    # blob_content = blob_data.decode('utf-8')
    print("Blob content:")

def get_data_from_blob(filename,container):
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get the container client
    container_client = blob_service_client.get_container_client(container)

    # Get the blob client (for the specific file)
    blob_client = container_client.get_blob_client(filename)
    # Read the blob's content into memory
    blob_data = blob_client.download_blob().readall()
    # Convert the content to string (assuming it's a text file)
    #blob_content = blob_data.decode('')
    #print("Blob content:")
    #print(blob_data)
    return blob_data

def getEmbeddingFiles():
    container_name = "metadafiles"

    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)
    blobs_list = container_client.list_blobs()
    metadata_file_dict = {}  # Initialize an empty dictionary
    i=0
    # Iterate through the blobs and print their names
    for blob in blobs_list:
        file_name = os.path.splitext(blob.name)[0]
        print(blob.name)  # Print the name of each blob
        i += 1
        metadata_file_dict[str(i)] = {
            'name': file_name
        }
    
    return metadata_file_dict  # Return the dictionary containing file details

