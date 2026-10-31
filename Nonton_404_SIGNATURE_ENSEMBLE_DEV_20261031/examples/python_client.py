"""Simple API client. Set NONTON_API_KEY in environment, then run after starting API."""
import os
import urllib.request
key=os.environ['NONTON_API_KEY']
req=urllib.request.Request('http://127.0.0.1:8080/v1/forecasts?limit=3',headers={'X-API-Key':key})
print(urllib.request.urlopen(req,timeout=30).read().decode())
