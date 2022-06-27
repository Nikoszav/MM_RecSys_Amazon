from cmath import phase
from models.model_baseline_cls import NeuralColabFilteringNet
import config

import numpy as np
import pandas as pd
import wandb

ratings = pd.read_pickle(config.ratings_file)
# ratings = pd.read_csv("D:/University/Edinburgh/Dissertation/Data/ml-20m/ratings.csv")
data = pd.read_pickle(config.data_file)

print("The dataset file is this:")
print(config.ratings_file)

user_count = ratings["userId"].nunique()
movie_count = data["movieId"].nunique()


# Let's peek into the loaded DataFrames
print("Sneak peek into the ratings DataFrame:\n\n", ratings.head(), "\n")
print(f"Number of users: {user_count:,}")
print(f"Number of ratings: {ratings.shape[0]:,}")
print(f'Number of rated movies: {ratings["movieId"].nunique():,}\n')
print(f"Sneak peek into the movies DataFrame:\n\n", data.head(), "\n")
print(
    f"Number of movies: {movie_count:,} - notice this is higher than the number of rated movies!"
)

print(
    "##########################################################################################"
)

# We marge the df so we can take the metadata for the movies
union = pd.merge(ratings, data, on="movieId", how="inner")
union.sort_values(by=["timestamp"], inplace=True)

# print (union.head())


# Define our input and labels data X,Y
X = union[["userId", "movieId", "timestamp"]]

# Because we change the problem and we treat it like classification

Y = union["rating"].astype(np.float32)
# Get one hot encoding of columns B
Y_one_hot = pd.get_dummies(Y)
# print(Y_one_hot.head())
# print("okay")

# print(X[0:10])

# data = Movies()
# labels = data.get_ratings()
# user_count = 900
# movie_count = 900
# print(test_x["rating"])
# print(test_x.__getitem__(0))


# Let's set the split ratio and run the split
from sklearn.model_selection import train_test_split

random_state = 7
test_size = 0.2

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_one_hot, test_size=test_size, shuffle=False
)
datasets = {"train": (X_train, Y_train), "test": (X_test, Y_test)}

# Checking the timestamps are right
# print(X_train.tail())
# print(X_test.head())

print(
    f"Training dataset sample size: {len(X_train):,} positive samples ({len(X_train)/len(X)*100:.0f}%)"
)
print(
    f"Test dataset sample size: {len(X_test):,} positive samples ({len(X_test)/len(X)*100:.0f}%)"
)
print(f"Total dataset sample size: {len(X):,} positive samples (100%)")

print(
    "##########################################################################################"
)

ncf = NeuralColabFilteringNet(user_count, movie_count, n_classes=len(Y.unique()))
print(f"Our model architecture:\n\n{ncf}\n")

# Let's look at the model size
num_params = sum(p.numel() for p in ncf.parameters())
print(
    f"Number of model parameters: {num_params:,}, model training size: {num_params*4/(1024**2):.2f} MB"
)

from training import train

# TODO: this will be added to the config file
hyperparameters = dict(
    lr=1e-5,
    wd=1e-4,
    batch_size=1020,
    max_epochs=50,
    early_stop_epoch_threshold=4,
    model="mm_baseline_cls",
)
wandb.init(project="MM_rec_sys", config=hyperparameters)


train(ncf, datasets, hyperparameters)
