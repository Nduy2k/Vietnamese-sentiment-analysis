from torch import nn
from torch.nn import Linear, ReLU, Softmax, Dropout
from transformers import AutoModel, AutoTokenizer

PRE_TRAINED_MODEL_NAME = "vinai/phobert-base"
NUM_CLASS = 4


class ClassifierModel(nn.Module):
    def __init__(self, bert, n_class=NUM_CLASS, drop_prob=0.3):
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
        return x


def load_bert(model_name=PRE_TRAINED_MODEL_NAME):
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
