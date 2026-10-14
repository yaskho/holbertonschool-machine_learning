#!/usr/bin/env python3
"""
Script to fetch and print the location of a GitHub user.
Handles rate limits and non-existent users.
"""
import requests
import sys
import time

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    url = sys.argv[1]
    res = requests.get(url)

    if res.status_code == 200:
        data = res.json()
        print(data.get('location'))
    elif res.status_code == 404:
        print("Not found")
    elif res.status_code == 403:
        reset_time = int(res.headers.get('X-Ratelimit-Reset', 0))
        current_time = int(time.time())
        minutes = int((reset_time - current_time) / 60)
        print("Reset in {} min".format(minutes))
