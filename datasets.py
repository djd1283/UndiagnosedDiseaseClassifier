from torch.utils.data import Dataset
import csv
import random
import torch
import math
import numpy as np
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from sentence_transformers import SentenceTransformer


class RedditUndiagnosedDataset(Dataset):
    def __init__(self, accepted_file, rejected_file, seed=1234):
        super().__init__()

        print('Loading pre-trained models')
        self.extractor = UndiagnosedFeatureExtractor()

        texts = []  # Reddit posts
        labels = []  # True if undiagnosed disease, false otherwise

        # we read through the accepted and rejected files and
        with open(accepted_file, 'r', newline='\n') as f_accepted:
            reader = csv.reader(f_accepted, delimiter='\t')
            for submission in reader:
                if len(submission[1]) > 0:
                    texts.append(submission[0] + ' ' + submission[1])
                    labels.append(True)

        with open(rejected_file, 'r', newline='\n') as f_rejected:
            reader = csv.reader(f_rejected, delimiter='\t')
            for submission in reader:
                if len(submission[1]) > 0:
                    texts.append(submission[0] + ' ' + submission[1])
                    labels.append(False)

        # shuffle examples into random order
        random.seed(seed)
        random.shuffle(texts)
        random.seed(seed)
        random.shuffle(labels)

        self.texts = texts
        self.labels = labels

        print('Extracting features')
        self.features = self.extractor.extract_features(self.texts)
        print(self.features.shape)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class UndiagnosedFeatureExtractor:
    def __init__(self):
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt').cuda()
        self.embedder = SentenceTransformer('bert-base-nli-mean-tokens').cuda()
        self.pos_phrase = "I have an undiagnosed disease. "
        self.keywords = [term.strip().lower() for term in open('tweet_crawler/terms.txt').read().split('\n')
                         if term != "" and term != "undiagnosed" and term != "disease"]
        # self.phrase_gpt_score = gpt_log_prob_score([self.phrase], self.gpt, self.tokenizer)
        self.pos_phrase_emb = self.embedder.encode([self.pos_phrase])[0]

    def extract_features(self, texts):

        # SBERT SIMILARITY FEATURE
        sbert_scores = sentence_bert_score(texts, [self.pos_phrase] * len(texts), self.embedder, return_all=True)

        # GPT LOG PROBABILITY FEATURES
        text_gpt_scores = gpt_log_prob_score(texts, self.gpt, self.tokenizer, return_all=True)
        pos_phrase_and_texts = [text + self.pos_phrase for text in texts]
        pos_phrase_text_gpt_scores = gpt_log_prob_score(pos_phrase_and_texts, self.gpt, self.tokenizer, return_all=True)
        # neg_phrase_text_gpt_scores = gpt_log_prob_score(neg_phrase_and_texts, self.gpt, self.tokenizer, return_all=True)

        phrase_text_mmis = []
        for pos_phrase_score, text_score in zip(pos_phrase_text_gpt_scores, text_gpt_scores):
            # negate loss for log probability
            phrase_text_mmi = (pos_phrase_score - text_score) / text_score
            phrase_text_mmis.append(phrase_text_mmi)

        # TEXT LENGTH FEATURE
        text_lens = [math.log(len(text.split())) for text in texts]

        # KEYWORD FEATURE
        texts_have_keywords = []
        for text in texts:
            text_has_keywords = False
            text_lower = text.lower()
            for keyword in self.keywords:
                if keyword in text_lower:
                    text_has_keywords = True

            texts_have_keywords.append(text_has_keywords)

        # DOCTORS FEATURE
        texts_have_doctors = ['doctors' in text.lower() for text in texts]
        print(texts_have_doctors)

        return np.array(list(zip(sbert_scores, text_gpt_scores, phrase_text_mmis, text_lens, texts_have_keywords,
                                 texts_have_doctors)))


def sentence_bert_score(r1, r2, embedder, return_all=False):
    """Compute cosine similarity between SBERT embeddings of corresponding sentences in r1 and r2."""
    with torch.no_grad():
        r1_embs = embedder.encode(r1)
        r2_embs = embedder.encode(r2)

        bert_scores = []
        for r1_emb, r2_emb in zip(r1_embs, r2_embs):
            bert_score = np.dot(r1_emb, r2_emb) / np.linalg.norm(r1_emb) / np.linalg.norm(r2_emb)
            bert_scores.append(bert_score)

    if return_all:
        return bert_scores
    else:
        return np.mean(bert_scores)


def gpt_log_prob_score(sentences, model, tokenizer, max_len=256, return_all=False):
    """Calculate the loss value of predicted sentences under a GPT model as a measure of fluency."""

    with torch.no_grad():
        losses = []
        for sentence in sentences:
            sentence = ' '.join(sentence.split()[:max_len])
            input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0).cuda()  # Batch size 1
            outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            losses.append(loss.item())

    if return_all:
        return losses
    else:
        return np.mean(losses)













