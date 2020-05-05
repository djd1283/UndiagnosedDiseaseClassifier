import praw
import csv
import os
from tqdm import tqdm

health_subreddits = ['diagnoseme', 'health', 'undiagnosed']  # order of subreddits we search

health_subreddit_keywords = \
{
    'undiagnosed':  {},
    'health':       {'undiagnosed', 'disease'},
    'diagnoseme':   {'undiagnosed'}
}

health_subreddit_limits = \
{
    'undiagnosed': 0,
    'health': 1000000,
    'diagnoseme': 1000000
}

neg_posts_multiplier = 10
data_dir = 'data/'
accepted_submissions_file = 'accepted_submissions.tsv'
rejected_submissions_file = 'rejected_submissions.tsv'
titles_file = 'submission_titles.txt'
start_over = False


def annotate_undiagnosed_reddit_posts(reddit, ignored_titles=None):
    accepted_submissions = []
    rejected_submissions = []

    for subreddit in health_subreddits:
        print(f'Searching subreddit {subreddit}')

        limit = health_subreddit_limits[subreddit]

        if limit > 0:

            num_posts_analyzed = 0

            bar = tqdm(total=limit)

            while num_posts_analyzed < limit:

                for submission in reddit.subreddit(subreddit).new(limit=limit):

                    bar.update(1)
                    num_posts_analyzed += 1

                    keywords = health_subreddit_keywords[subreddit]



                    # things to save
                    title = submission.title
                    text = ' '.join(submission.selftext.split())

                    if ignored_titles is not None:
                        if title in ignored_titles:
                            continue

                    contains_all_keywords = True
                    for keyword in keywords:
                        if keyword not in title and keyword not in text:
                            contains_all_keywords = False

                    if not contains_all_keywords:
                        continue

                    print('Subreddit: %s' % subreddit)
                    print('Title:%s' % submission.title)
                    print('Body: %s' % submission.selftext)

                    accept = input('Accept? (y/n/quit):')

                    if accept.lower() == 'quit' or accept.lower() == 'q':
                        return accepted_submissions, rejected_submissions

                    if accept.lower() == 'yes' or accept.lower() == 'y':
                        accepted_submissions.append([title, text, subreddit])
                    else:
                        rejected_submissions.append([title, text, subreddit])



    return accepted_submissions, rejected_submissions


def load_credentials():
    # import credentials from creds.txt in this directory
    with open('creds.txt', 'r') as f:
        text = f.read()
        lines = text.split('\n')
        creds = {line.split(': ')[0].strip(): line.split(': ')[1].strip() for line in lines if line != ''}
        print(creds)
    return creds





def main():

    creds = load_credentials()

    reddit = praw.Reddit(client_id=creds['client'],
                         client_secret=creds['secret'],
                         user_agent=creds['agent'])

    print(reddit.read_only)  # Output: True

    global start_over

    titles_path = os.path.join(data_dir, titles_file)

    ignored_titles = []
    if not start_over and os.path.exists(titles_path):
        ignored_titles = set(open(titles_path, 'r').read().split('\n')[:-1])
    else:
        start_over = True  # force us to start over if we don't have this file

    print('Number of previously saved submissions: %s' % len(ignored_titles))

    accepted_submissions, rejected_submissions = annotate_undiagnosed_reddit_posts(reddit, ignored_titles=ignored_titles)

    # negative_submissions = generate_negative_posts(len(accepted_submissions) * neg_posts_multiplier)

    if len(accepted_submissions) > 0:
        print(accepted_submissions[0][0])

    os.makedirs(data_dir, exist_ok=True)

    # here we write titles to check for duplicates
    with open(os.path.join(data_dir, titles_file), 'w' if start_over else 'a') as f_titles:
        for submission in (accepted_submissions + rejected_submissions):
            f_titles.write(submission[0] + '\n')

    # we write posts annotated as undiagnosed disease to accepted file
    with open(os.path.join(data_dir, accepted_submissions_file), 'w' if start_over else 'a', newline='\n') as f_accepted:
        writer = csv.writer(f_accepted, delimiter='\t')
        for submission in accepted_submissions:
            writer.writerow(submission)

    # posts we annotated as not undiagnosed disease go to rejected file
    with open(os.path.join(data_dir, rejected_submissions_file), 'w' if start_over else 'a', newline='\n') as f_rejected:
        writer = csv.writer(f_rejected, delimiter='\t')
        for submission in rejected_submissions:
            writer.writerow(submission)


if __name__ == '__main__':
    main()