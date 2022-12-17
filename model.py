import torch
from torch import nn
from torch.nn import Dropout, Linear, ReLU, Softmax, LogSoftmax
from transformers import AutoModel, AutoTokenizer

PRE_TRAINED_MODEL_NAME = "vinai/phobert-base"
NUM_CLASS = 4


class ClassifierModel(nn.Module):
    def __init__(self, bert, n_class: int = NUM_CLASS, drop_prob: float = 0.3):
        """
        :param bert: model BERT to extract features
        :param n_class: (int) number of class
        :param drop_prob: drop prob
        """
        super().__init__()

        self.bert = bert
        self.drop_prob = drop_prob
        self.n_class = n_class

        # Fully Connected Layers
        self.fc1 = Linear(in_features=768, out_features=512)
        self.fc2 = Linear(in_features=512, out_features=self.n_class)

        # Activate Function Layers
        self.relu = ReLU()
        self.soft_max = Softmax(dim=1)

        # Dropout Layer
        self.drop_out = Dropout(self.drop_prob)

    def forward(self, sentences_id, attention_mask):
        _, pooled_out = self.bert(sentences_id, attention_mask, return_dict=False)
        # pooled_out size: [batch_size, 768]
        x = self.drop_out(pooled_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        # x = self.soft_max(x)
        return x


class ClassifierModel1(nn.Module):
    """
    Arguments:
        bert (model): model BERT to extract features
        n_class (int): number of class
    """

    def __init__(self, bert, n_class, drop_prob=0.1):
        super().__init__()

        self.bert = bert
        self.drop_prob = drop_prob
        self.n_class = n_class

        # Fully Connected Layers
        self.fc1 = Linear(in_features=768, out_features=512, bias=True)
        self.fc2 = Linear(in_features=512, out_features=self.n_class, bias=True)

        # Activate Function Layers
        self.relu = ReLU()
        self.softmax = LogSoftmax(dim=1)

        # Dropout Layer
        self.drop_out = Dropout(self.drop_prob)

    def forward(self, sentences_id, attention_mask):
        _, pooled_out = self.bert(sentences_id, attention_mask, return_dict=False)
        # pooled_out size: [batch_size, 768]
        # x = self.drop_out(pooled_out)
        x = self.fc1(pooled_out)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


def load_bert(model_name: str = PRE_TRAINED_MODEL_NAME):
    """
    Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese
    (Pho, i.e. "Phở", is a popular food in Vietnam)
    Pre-trained name:
        PhoBERT Base:  "vinai/phobert-base"
        PhoBERT Large: "vinai/phobert-large"
    """
    phobert = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, phobert
