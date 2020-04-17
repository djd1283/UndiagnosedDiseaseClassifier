from torch.utils.data import Dataset
import csv
import random


class RedditUndiagnosedDataset(Dataset):
    def __init__(self, accepted_file, rejected_file, seed=1234):
        super().__init__()

        texts = []  # Reddit posts
        labels = []  # True if undiagnosed disease, false otherwise

        # we read through the accepted and rejected files and
        with open(accepted_file, 'r', newline='\n') as f_accepted:
            reader = csv.reader(f_accepted, delimiter='\t')
            for submission in reader:
                texts.append(submission)
                labels.append(True)

        with open(rejected_file, 'r', newline='\n') as f_rejected:
            reader = csv.reader(f_rejected, delimiter='\t')
            for submission in reader:
                texts.append(submission)
                labels.append(False)

        # shuffle examples into random order
        random.seed(seed)
        random.shuffle(texts)
        random.seed(seed)
        random.shuffle(labels)

        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]













