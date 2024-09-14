# %%
import requests
import json

def push_message(title, body):
    token = "o.dXbWhpbjCdSzCrXPjbwXfg4roPLcZHy5"
    url = "https://api.pushbullet.com/v2/pushes"
    headers = {"content-type": "application/json", "Authorization": 'Bearer '+token}
    data_send = {"type": "note", "title": title, "body": body}

    _r = requests.post(url, headers=headers, data=json.dumps(data_send))

if __name__ == "__main__":
    push_message("test", "test")
# %%
