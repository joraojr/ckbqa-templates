from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score
import torch


class Metrics():
    def __init__(self):
        pass

    def accuracy_score(self, predictions, labels, vocab_output):
        labels = torch.tensor([vocab_output.getIndex(str(int(label))) for label in labels], dtype=torch.float)
        correct = (predictions == labels).sum()
        total = labels.size(0)
        acc = float(correct) / total
        return acc

    def balanced_accuracy(self, predictions, labels, vocab_output):
        labels = torch.tensor([vocab_output.getIndex(str(int(label))) for label in labels], dtype=torch.float)

        return balanced_accuracy_score(labels, predictions)
