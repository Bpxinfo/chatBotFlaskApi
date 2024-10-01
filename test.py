import requests
import json

url = "http://127.0.0.1:5000/chat"
headers = {"Content-Type": "application/json"}
data = {
    "message": "What is the capital of France?"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())

#testing to see if api is running correctly