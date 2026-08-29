import requests

API_TOKEN = "q2fTevl8Dy7XUpDtHOxG1PK5olM5fB"  # Replace with your actual MVSEP API key
BASE_URL = "https://mvsep.com/api"

# Test getting separation types
print("Testing GET /separation/get_types ...")
resp = requests.get(
    f"{BASE_URL}/separation/get_types",
    params={'api_token': API_TOKEN}
)

print(f"HTTP Status: {resp.status_code}")
print(f"Raw response (first 300 chars): {resp.text[:300]}")

if resp.status_code == 200:
    try:
        data = resp.json()
        print("Success! Received types.")
        # Print first few types for sanity
        if 'data' in data and 'types' in data['data']:
            types = data['data']['types']
            print(f"Number of types: {len(types)}")
            for t in types[:5]:
                print(f"  {t['id']}: {t['name']}")
        else:
            print("Unexpected response structure:", data)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
else:
    print(f"Request failed with status {resp.status_code}")
