import requests
import time
import os

API_TOKEN = "q2fTevl8Dy7XUpDtHOxG1PK5olM5fB"  # Your real token
BASE_URL = "https://mvsep.com/api"
AUDIO_FILE = "/home/fritz/I_Saw_The_Light.mp3"  # Replace with a real audio file path

# 1. Create a separation job
print("Creating separation job...")
create_url = f"{BASE_URL}/separation/create"
files = {'audiofile': open(AUDIO_FILE, 'rb')}
data = {
    'api_token': API_TOKEN,
    'sep_type': '123',  # BS PolarFormer (124 bands)
}

resp = requests.post(create_url, data=data, files=files)
files['audiofile'].close()
print(f"Create response status: {resp.status_code}")
print(f"Create response: {resp.text}")

if resp.status_code != 200:
    print("Creation failed.")
    exit()

result = resp.json()
if not result.get('success'):
    print(f"API error: {result}")
    exit()

# CORRECTED: hash is inside 'data'
job_hash = result['data']['hash']
print(f"Job hash: {job_hash}")

# 2. Poll for status
print("\nPolling for status...")
status_url = f"{BASE_URL}/separation/get"
while True:
    # CORRECTED: include both hash and api_token
    status_resp = requests.get(status_url, params={'hash': job_hash, 'api_token': API_TOKEN})
    status_data = status_resp.json()
    status = status_data.get('status')
    print(f"Status: {status}")
    if status == 'done':
        files_data = status_data.get('data', {}).get('files', [])
        print(f"Download URLs: {files_data}")
        break
    elif status in ['waiting', 'processing', 'distributing', 'merging']:
        time.sleep(10)
    elif status == 'failed':
        print(f"Job failed: {status_data}")
        break
    else:
        print(f"Unexpected status: {status_data}")
        break
