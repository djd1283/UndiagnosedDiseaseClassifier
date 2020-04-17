"""Get random submissions from Reddit as negative examples for classifier."""
import os
import csv
import praw
from filter_reddit_data import load_credentials
from tqdm import tqdm


data_dir = 'data/'
negative_submissions_file = 'negative_submissions.tsv'
accepted_submissions_file = 'accepted_submissions.tsv'


def generate_negative_posts(reddit, n_posts):
    # TODO find regular posts from popular subreddits as negative examples for training a classifier

    negative_posts = []

    bar = tqdm(total=n_posts)

    while True:
        random_submissions = reddit.subreddit('all').random_rising(limit=n_posts)
        random_submissions = [submission for submission in random_submissions]
        for submission in random_submissions:
            title = submission.title
            text = ' '.join(submission.selftext.split())
            subreddit = submission.subreddit
            if len(text) > 0:
                negative_posts.append([title, text, subreddit])
                bar.update()
                if len(negative_posts) >= n_posts:
                    return negative_posts


def main():

    creds = load_credentials()

    reddit = praw.Reddit(client_id=creds['client'],
                         client_secret=creds['secret'],
                         user_agent=creds['agent'])

    # we generate the same number of negative examples as positive examples
    n_posts = sum([1 for line in open(os.path.join(data_dir, accepted_submissions_file), 'r')])
    print(f'Number of negative submissions: {n_posts}')

    negative_submissions = generate_negative_posts(reddit, n_posts)

    # random posts from common subreddit we use as negative examples and save to negative file
    with open(os.path.join(data_dir, negative_submissions_file), 'w', newline='\n') as f_negative:
        writer = csv.writer(f_negative, delimiter='\t')
        for submission in negative_submissions:
            writer.writerow(submission)


if __name__ == '__main__':
    main()


