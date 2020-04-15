import json
from tqdm import tqdm
import pickle
import os
import csv

submissions = []

dump_files = ['data/RS_2015-08', 'data/RS_2015-09']
save_file = 'data/dump_keyword_submissions.tsv'

start_over = False

for dump_file in dump_files:
    print(f'Analyzing file: {dump_file}')
    with open(dump_file, 'r') as f:
        for line in tqdm(f):
            submission = json.loads(line)
            # print(submission)
            # print(submission['title'])
            # print(submission['selftext'])

            if 'undiagnosed' in submission['title'] or 'undiagnosed' in submission['selftext']:
                if 'disease' in submission['title'] or 'disease' in submission['selftext']:
                    title = submission['title']
                    text = ' '.join(submission['selftext'].split())
                    subreddit = submission['subreddit']
                    submissions.append([title, text, subreddit])

# posts we annotated as not undiagnosed disease go to rejected file
with open(save_file, 'w' if start_over else 'a', newline='\n') as f:
    writer = csv.writer(f, delimiter='\t')
    for submission in submissions:
        writer.writerow(submission)
