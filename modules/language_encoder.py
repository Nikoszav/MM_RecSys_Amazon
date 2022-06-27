import config
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Movie_Bert(nn.Module):
    def __init__(self, h_dim=None):
        super(Movie_Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.encoder = BertModel.from_pretrained("bert-base-cased")

    # maybe will be a prblem with this being a batch
    def forward(self, input, maxLength=177):
        tokenizer_output = self.tokenizer(
            input,
            max_length=maxLength,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens, masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )
        # movie_text_tokens = movie_text_tokens.squeeze()
        # movie_text_mask = movie_text_mask.squeeze()
        _, x = self.encoder(input_ids=tokens, attention_mask=masks, return_dict=False)
        return x


def get_language_encoder(encoder_source="BERT", h_dim=None):
    return Movie_Bert(h_dim=None)
