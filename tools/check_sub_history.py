import json
import urllib.request
from pathlib import Path

cred_file = Path(r"C:\Users\Cyrus\.kaggle\credentials.json")
creds = json.loads(cred_file.read_text(encoding="utf-8"))
access_token = creds["access_token"]

url = "https://www.kaggle.com/api/v1/competitions/rsna-knee-abnormality-detection/submissions"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f"Total submissions returned: {len(data)}")
        for i, s in enumerate(data[:10]):
            date = s.get("submitted", "")[:19]
            status = s.get("status", "")
            score = s.get("publicScore", "N/A")
            desc = s.get("description", "")[:60]
            err = s.get("errorDescription", "")
            print(f"[{i+1}] {date} | Status: {status} | Score: {score} | Desc: {desc} | Err: {err}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode()}")
except Exception as e:
    print(f"Request failed: {e}")
