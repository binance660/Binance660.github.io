#!/usr/bin/env python3
import os, hmac, hashlib, requests, urllib.parse, time

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

def sign_params(params):
    safe_params = {k: str(params[k]) for k in params}
    query = urllib.parse.urlencode(sorted(safe_params.items()))
    signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + "&signature=" + signature

try:
    params = {"timestamp": int(time.time() * 1000)}
    signed = sign_params(params)
    headers = {"X-MBX-APIKEY": API_KEY}
    r = requests.get(f"https://api.binance.com/api/v3/account?{signed}", headers=headers, timeout=10)
    if r.status_code == 200:
        print("✅ API KEYS VALID - Account accessible")
        data = r.json()
        print(f"   Trading Enabled: {data.get('canTrade', False)}")
        print(f"   Withdrawals Enabled: {data.get('canWithdraw', False)}")
    else:
        print(f"❌ ERROR: {r.status_code}")
        print(f"   Response: {r.text[:300]}")
except Exception as e:
    print(f"❌ EXCEPTION: {str(e)[:300]}")
