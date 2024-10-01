from azure.storage.fileshare import ShareServiceClient,generate_file_sas, ShareFileClient,FileSasPermissions
from io import BytesIO
import os
from flask import jsonify
from datetime import datetime, timedelta
import faiss
import requests
from dotenv import load_dotenv

load_dotenv()

account_key = os.getenv("AZURE_FILESHARE_ACCOUNTKEY")
# Azure File Share connection string
connection_string = os.getenv("AZURE_FILESHARE_CONNECTIONSTRING")

# Create the ShareServiceClient object
share_service_client = ShareServiceClient.from_connection_string(connection_string)

# Name of the file share (create one if it doesn't exist)
share_name = "ragchatfiles"
share_client = share_service_client.get_share_client(share_name)

# Name of the directory (optional), set to root ('/') if not using directories
directory_name = "ragfiles"
directory_client = share_client.get_directory_client(directory_name)


#check files exists or not
def checkFileInAzure(filename):
	files = directory_client.list_directories_and_files(filename)
	if not any(files):
		return 'yes'
	else:
		return "EXISTS"

# List files in the directory and print file name with last modified timestamp
def getFiles():
    files = directory_client.list_directories_and_files()
    file_dict = {}  # Initialize an empty dictionary

    for file in files:
        file_name = file['name']
        last_modified = file['last_modified']
        
        # Access the file size
        # You may need to get the file client to fetch the properties
        file_client = directory_client.get_file_client(file_name)
        properties = file_client.get_file_properties()  # Fetch file properties
        file_size = properties['size']  # Get the file size

        # Store the details in the dictionary
        file_dict[file_name] = {
            'last_modified': last_modified,
            'size': file_size
        }
    
    return file_dict  # Return the dictionary containing file details

def uploadFile_inazure(file):
	# Create a file client
    file_name = file.filename
    file_client = ShareFileClient.from_connection_string(
        conn_str=connection_string,
        share_name=share_name,
        file_path=f"{directory_name}/{file_name}"
    )

    try:
        # Upload the file
        file_client.upload_file(file)
        return jsonify({"message": f"File '{file_name}' uploaded successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def upload_EmbeddingFile(file_content, remote_file_name,dir_name):

    file_client = ShareFileClient.from_connection_string(
        conn_str=connection_string,
        share_name="files",
        file_path=f"{dir_name}/{remote_file_name}"
    )
    # Upload the file content (bytes) to Azure File Share
    file_stream = BytesIO(file_content)
    file_client.upload_file(file_stream)
    
    print(f"Uploaded {remote_file_name} to Azure File Share")

def getEmbeddingFiles():
    share_name = "files"
    share_client = share_service_client.get_share_client(share_name)

    # Name of the directory (optional), set to root ('/') if not using directories
    directory_name = "metadatafiles"
    directory_client = share_client.get_directory_client(directory_name)

    files = directory_client.list_directories_and_files()
    metadata_file_dict = {}  # Initialize an empty dictionary
    i=0
    for file in files:
        file_name = os.path.splitext(file['name'])[0]     
        # Store the details in the dictionary
        i += 1
        metadata_file_dict[str(i)] = {
            'name': file_name
        }
    
    return metadata_file_dict  # Return the dictionary containing file details

###############################################################
