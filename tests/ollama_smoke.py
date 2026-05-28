import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "mistral",
        "messages": [
            {
                "role": "user",
                "content": "hello"
            }
        ],
        "stream": False,
    },
)

print(response.status_code)
print(response.text)
