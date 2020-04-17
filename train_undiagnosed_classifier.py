"""This script uses the available accepted and rejected submissions labeled by hand to creaete classifier
between them."""
from datasets import RedditUndiagnosedDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import wandb
import numpy as np


accepted_submissions_file = 'data/accepted_submissions.tsv'
negative_submissions_file = 'data/negative_submissions.tsv'
rejected_submissions_file = 'data/rejected_submissions.tsv'
n_features = 1
n_epochs = 1000
learning_rate = 0.001
batch_size = 2


def main():
    wandb.init(project='undiagnosed_classifer', allow_val_change=True, dir='data/')

    # load dataset
    ds = RedditUndiagnosedDataset(accepted_file=accepted_submissions_file, rejected_file=negative_submissions_file)
    num_train_examples = round(len(ds) * 0.7)
    train_ds, val_ds = random_split(ds, [num_train_examples, len(ds) - num_train_examples])
    print(f'Dataset size: {len(ds)}')
    # load models required
    classifier = nn.Linear(n_features, 1)
    # load optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
    bce = nn.BCEWithLogitsLoss()
    # create dataloader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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


    # evaluate accuracy on validation set


if __name__ == '__main__':
    main()


