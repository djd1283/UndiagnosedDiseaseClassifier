"""This script uses the available accepted and rejected submissions labeled by hand to creaete classifier
between them."""
from datasets import RedditUndiagnosedDataset, TwitterUndiagnosedDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import wandb
import numpy as np


accepted_submissions_file = 'data/accepted_submissions.tsv'
negative_submissions_file = 'data/negative_submissions.tsv'
rejected_submissions_file = 'data/rejected_submissions.tsv'

train_accepted_tweets_file = 'data/tweet_dataset/train_selected.txt'
train_negative_tweets_file = 'data/tweet_dataset/train_texts_negative_clean2.txt'
val_accepted_tweets_file = 'data/tweet_dataset/val_selected.txt'
val_negative_tweets_file = 'data/tweet_dataset/val_texts_negative_clean2.txt'

trec_eval_labels_file = 'data/trec_eval_labels.txt'
trec_eval_scores_file = 'data/trec_eval_scores.txt'

n_features = 7
n_epochs = 1000
learning_rate = 0.001
batch_size = 2
seed = 1234
max_negatives = 100

# for predictable experiments
torch.random.manual_seed(seed)

dataset_name = "twitter"


def produce_files_for_rank_evaluation(test_ds, classifier):
    """The goal of this function is to produce ranking scores for all examples in the test dataset
    along with labels as to whether or not they contain an undiagnosed disease. Then save the labels and
    scores to different files according to trec_eval and use trec_eval to evaluate"""

    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ranking_scores = []
    labels = []

    for batch in test_dl:
        # train on examplesw
        with torch.no_grad():
            x = batch[0].float()
            y = batch[1].float().unsqueeze(1)
            preds = classifier(x)

        ranking_scores.append(preds)
        labels.append(y)

    ranking_scores = torch.cat(ranking_scores, 0).squeeze(1).numpy()
    labels = torch.cat(labels, 0).squeeze(1).numpy()

    with open(trec_eval_labels_file, 'w') as f_labels:
        with open(trec_eval_scores_file, 'w') as f_scores:
            for i, (score, label) in enumerate(zip(ranking_scores, labels)):
                # write label to one file, write score to another
                    f_labels.write(f'Q1046 0 {i} {label}\n')
                    f_scores.write(f'Q1046 0 {i} - {score} standard\n')






def main():
    wandb.init(project='undiagnosed_classifer', allow_val_change=True, dir='data/')

    # load dataset
    if dataset_name == "reddit":
        ds = RedditUndiagnosedDataset(accepted_file=accepted_submissions_file, rejected_file=negative_submissions_file,
                                      seed=seed)
        num_train_examples = round(len(ds) * 0.7)
        train_ds, val_ds = random_split(ds, [num_train_examples, len(ds) - num_train_examples])
    else:
        train_ds = TwitterUndiagnosedDataset(accepted_file=train_accepted_tweets_file, rejected_file=train_negative_tweets_file,
                                             seed=seed)
        val_ds = TwitterUndiagnosedDataset(accepted_file=val_accepted_tweets_file, rejected_file=val_negative_tweets_file,
                                           seed=seed)

        # the test set contains a larger number of negative examples (max_negatives) for top-k ranking
        test_ds = TwitterUndiagnosedDataset(accepted_file=val_accepted_tweets_file,
                                            rejected_file=val_negative_tweets_file,
                                            seed=seed, equal_accept_rej=False, max_examples=max_negatives)

    print(f'Train dataset size: {len(train_ds)}')
    print(f'Validation dataset size: {len(val_ds)}')
    # load models required
    classifier = nn.Linear(n_features, 1)
    # load optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
    bce = nn.BCEWithLogitsLoss()
    # create dataloader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # here we perform evaluation without training to test
    produce_files_for_rank_evaluation(test_ds, classifier)
    exit()

    # run through examples

    for epoch_idx in range(n_epochs):
        train_losses = []
        train_accuracies = []
        for batch in train_dl:
            # train on examples
            x = batch[0].float()
            y = batch[1].float().unsqueeze(1)

            optimizer.zero_grad()
            preds = classifier(x)
            loss = bce(preds, y)
            loss.backward()
            optimizer.step()

            acc = ((preds > 0).float() == y).float().mean()
            train_accuracies.append(acc)

            train_losses.append(loss.item())

        val_losses = []
        val_accuracies = []
        for batch in val_dl:
            with torch.no_grad():
                x = batch[0].float()
                y = batch[1].float().unsqueeze(1)

                preds = classifier(x)
                loss = bce(preds, y)
                val_losses.append(loss.item())

                acc = ((preds > 0).float() == y).float().mean()
                val_accuracies.append(acc)

        wandb.log({'train loss': np.mean(train_losses), 'val loss': np.mean(val_losses),
                   'train accuracy': np.mean(train_accuracies), 'val accuracy': np.mean(val_accuracies)})

    print(classifier.weight)
    print(classifier.bias)

    # evaluate performance metrics on validation set because we did not use it for hyperparameters
    # here we must grab all positive examples and a larger number of negative examples
    # save them to files of a format corresponding to the 'trec_eval' package

    # TODO produce ranking scores for each example in the test set, write them to files for trec_eval

    produce_files_for_rank_evaluation(test_ds, classifier)


if __name__ == '__main__':
    main()


