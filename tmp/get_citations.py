import os

import requests
import sys
import time

sleep_time = 20


def query_api(url, session):
    global sleep_time
    time.sleep(sleep_time / 1000.0)
    r = session.get(url)
    while r.status_code == 429:
        sleep_time *= 2
        print(
            f'WARNING: Hit rate limit. Increasing sleep to {sleep_time} ms',
            file=sys.stderr,
        )
        time.sleep(sleep_time / 1000.0)
        r = session.get(url)
    if r.status_code != 200:
        print(f'WARNING: Could not access url {url}', file=sys.stderr)
        return None
    else:
        return r.json()


# with open('s2key.txt', 'r') as f:
#   s2_key = next(f).strip()
session = requests.Session()


# session.headers.update({'x-api-key': s2_key})

def print_all_citations(conf, num_papers, save_to=None):
    if save_to is None:
        save_to = os.path.join('data', f'{conf}.txt')
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    if os.path.exists(save_to):
        with open(save_to) as f:
            try:
                for line in f:
                    pass
                start_pid, cites = line.strip().split()
                start_pid = int(start_pid)
                cites = int(cites)
            except:
                print("WARNING: Could not parse last line of file, overwriting it")
                start_pid = 1
    else:
        start_pid = 1
    with open(save_to, 'a+') as f:
        for pid in range(start_pid, num_papers + 1):
            aclid = f'{conf}.{pid}'
            s2url = f'https://api.semanticscholar.org/v1/paper/ACL:{aclid}'
            paper_data = query_api(s2url, session)
            if paper_data != None:
                citations = len(paper_data['citations'])
                print(f'{aclid}\t{citations}')
                f.write(f'{aclid}\t{citations}\n')
                f.flush()


# EMNLP Papers
print_all_citations('2020.emnlp-main', 752)
print_all_citations('2020.findings-emnlp', 447)
print_all_citations('2021.emnlp-main', 752)
print_all_citations('2021.findings-emnlp', 447)