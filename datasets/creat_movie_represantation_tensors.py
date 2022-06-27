import torch
from torch.utils.data import DataLoader


from modules.poster_encoder import get_feature_extractor
from datasets.datasets import Movies


def movie_represantation(movies_dataset):
    for movie in movies_dataset:
        movie_id, movie_image_encoder, movie_rep = movie
        # make the represantation of the poster
        image_extractor = get_feature_extractor()
        movie_image_encoder = image_extractor(
            movie_image_encoder
        )  # [batch_size, 512] -> Output of resnet
        # create paths for the tensors to be stored
        cnn_filepath = (
            "D:/University/Edinburgh/Dissertation/data/movie_represantation/cnn_output/"
            + str(movie_id.item())
            + ".pt"
        )
        bert_filepath = (
            "D:/University/Edinburgh/Dissertation/data/movie_represantation/bert_output/"
            + str(movie_id.item())
            + ".pt"
        )
        # save tensor
        torch.save(movie_image_encoder, cnn_filepath)
        torch.save(movie_rep, bert_filepath)


if __name__ == "__main__":
    print("running")
    movies_dataset = DataLoader(Movies(), batch_size=1)
    movie_represantation(movies_dataset)
