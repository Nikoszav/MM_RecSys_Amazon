import config
import torch 
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

class BertModel(nn.Module):
    def __init__(self, h_dim=None):
        super(BertModel, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
    
    #maybe will be a prblem with this being a batch 
    def forward(self, input, maxLength=177): 
        tokenizer_output = self.tokenizer(input, max_length=maxLength, padding="max_length", truncation=True, return_tensors="pt")
        tokens, masks = tokenizer_output['input_ids'], tokenizer_output['attention_mask']
        # movie_text_tokens = movie_text_tokens.squeeze()
        # movie_text_mask = movie_text_mask.squeeze()
        x = self.encoder(input_ids=tokens, attention_mask=masks)[0]
        return x 
    
# class BertEncoder(nn.Module):
#     def __init__(self, h_dim=None):
#         super(BertEncoder, self).__init__()
#         self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
#         if config.train_bert is False:      # fine tune bert or not 
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#         # If mapping to a hidden dimension size, otherwise output will be 768
#         if h_dim:
#             self.map_to_hid = True
#             self.to_hid = nn.Linear(768, h_dim)
#         else:
#             self.map_to_hid = False

#     def forward(self, tokens, masks=None):
#         # tokens = torch.tensor([[ 101, 2054, 4338, 2003, 1996, 6847, 2835, 1029,  102,    0,    0,    0, 0,    0]])
#         # masks = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
#         if config.train_bert:
#             x = self.encoder(input_ids=tokens, attention_mask=masks)[0]

#         else:
#             self.encoder.eval()
#             with torch.no_grad():
#                 x = self.encoder(input_ids=tokens, attention_mask=masks)[0]

#         if self.map_to_hid:
#             x = self.to_hid(x)

#         return x

# def get_language_encoder(encoder_source='BERT', h_dim=None):
#     return BertEncoder(h_dim=None)