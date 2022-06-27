import torch
import torch.nn as nn

from modules.language_encoder import get_language_encoder

from modules.poster_encoder import get_feature_extractor

import config


class NeuralColabFilteringNet(nn.Module):
    """
    Creates an NCF (Neural Collaborative Filtering) network, with configurable model architecture
    
    Args:
    user_count(int): Number of unique users in the dataset
    movie_count(int): Number of unique movies in the dataset
    embedding_size(int)[Optional]: Size of the user and movie embedding, defaults to 32
    hidden_layers(tuple)[Optional]: Tuple of integers defining the number of hidden MLP layers and the number of units in each layer, defaults to (64,32,16,8)
    dropout_rate(float)[Optional]: Dropout rate to apply after each layer in the range of [0 1], defaults to None
    output_range(tuple)[Optional]: Represents the output range, defaults to (1 5) per the star ratings
    """

    def __init__(
        self,
        user_count,
        movie_count,
        embedding_size=300,
        hidden_layers=(1580, 128, 64, 32, 16, 8),
        dropout_rate=0.3,
        output_range=(1, 5),
    ):
        super().__init__()

        # Initialize embedding hash sizes
        self.user_hash_size = user_count
        # self.movie_hash_size = movie_count

        # Initialize the model architecture components
        self.user_embedding = nn.Embedding(user_count, embedding_size)
        # self.movie_embedding = nn.Embedding(movie_count, embedding_size)
        # self.poster_hd = nn.Linear(768, 768)
        # self.poster_hd = nn.Linear(5)

        self.MLP = self._gen_MLP(embedding_size, hidden_layers, dropout_rate)
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

        # Initialize output normalization parameters
        assert (
            output_range and len(output_range) == 2
        ), "output_range has to be a tuple with two integers"
        self.norm_min = min(output_range)
        self.norm_range = abs(output_range[0] - output_range[1]) + 1

        self._init_params()

    def _gen_MLP(self, embedding_size, hidden_layers_units, dropout_rate):
        "Generates the MLP portion of the model architecture"

        assert (1580) == hidden_layers_units[
            0
        ], "First input layer number of units has to be equal to twice the embedding size!"

        hidden_layers = []
        input_units = hidden_layers_units[0]
        # print(input_units)
        for num_units in hidden_layers_units[1:]:
            hidden_layers.append(nn.Linear(input_units, num_units))
            hidden_layers.append(nn.ReLU())
            if dropout_rate:
                hidden_layers.append(nn.Dropout(dropout_rate))
            input_units = num_units

        hidden_layers.append(nn.Linear(hidden_layers_units[-1], 1))
        hidden_layers.append(nn.Sigmoid())
        return nn.Sequential(*hidden_layers)

    def _init_params(self):
        "Initializes model parameters"

        def weights_init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        # self.movie_embedding.weight.data.uniform_(-0.05, 0.05)
        self.MLP.apply(weights_init)

    def forward(self, user_id, movie_id):
        "Computes forward pass"
        user_features = self.user_embedding(user_id % self.user_hash_size)
        for i, id in enumerate(movie_id.cpu().numpy()):
            if i == 0:
                bert_output = torch.load(config.bert_filepath + str(id) + ".pt")
                cnn_output = torch.load(config.cnn_path + str(id) + ".pt").unsqueeze(
                    dim=0
                )

            else:
                bert_output = torch.cat(
                    (bert_output, torch.load(config.bert_filepath + str(id) + ".pt"))
                )
                cnn_output = torch.cat(
                    (
                        cnn_output,
                        torch.load(config.cnn_path + str(id) + ".pt").unsqueeze(dim=0),
                    )
                )

        bert_output = bert_output.squeeze(dim=1)

        movie_features = torch.cat((bert_output, cnn_output), dim=1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        movie_features = movie_features.to(device)
        # print(movie_features.shape)

        x = torch.cat([user_features, movie_features], dim=1)
        # print(x.shape)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.MLP(x)
        normalized_output = x * self.norm_range + self.norm_min
        return normalized_output
