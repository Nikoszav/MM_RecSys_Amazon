
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageFile
from modules.language_encoder_concat import BertModel
from modules.poster_encoder import get_feature_extractor 

import pandas as pd
import config
import os
import numpy as np
import pickle
from transformers import DeiTFeatureExtractor
import ast
import random 
import math

class Movies(Dataset):
    """Movies dataset."""

    def __init__(self):
        """
        Args:
            
        """
        self.poster_folder_path = config.poster_folder_path
        self.movies_dataset = pd.read_pickle(config.data_file)
        self.rating_dataset = pd.read_pickle(config.ratings_file)
        union = pd.merge(self.movies_dataset, self.rating_dataset, on='movieId', how='inner')
        # print (union)
        # Define our input and labels data X,Y
        self.movies_ratings_dataset = union[['userId','movieId', 'rating','overview', "title", "path" ]]
        self.ratings = union['rating'].astype(np.float32)
        self.resize = transforms.Resize(config.poster_size)
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #not sure why this normalization
        self.to_tensor = transforms.ToTensor()
        
        self.encoder = BertModel()
        
        self.image_encoder = get_feature_extractor()

    def get_ratings(self):
      return list(self.movies_ratings_dataset["rating"])
    
    def __len__(self):
        return len(self.movies_ratings_dataset)

    def __getitem__(self, idx):

        sample = self.movies_ratings_dataset.iloc[idx]
        # user_id = sample["userId"]
        # movie_id = sample["movieId"]
        movie_descr = str(sample["title"]) + " " + str(sample["overview"])
        poster_name = os.path.join(self.poster_folder_path,
                                sample["path"])

        #not sure if this is the correct path 
        image = Image.open(poster_name).convert('RGB')  #read and convert images to RGB

        image_tensor = self.norm(self.to_tensor(self.resize(image)))    #normalization and resize of poster
        
        # movie_image_encoder = self.image_encoder(image_tensor)
        
        movie_rep = self.encoder(movie_descr)   #tokenization of the movie description

        return sample["userId"],sample["movieId"], image_tensor, movie_rep
    
    
    

class DatasetBatchIterator():
  "Iterates over labaled dataset in batches"
  def __init__(self, X, Y, batch_size, shuffle=True):
    self.X = np.asarray(X)
    self.Y = np.asarray(Y)
    
    if shuffle:
      index = np.random.permutation(X.shape[0])
      X = self.X[index]
      Y = self.Y[index]

    self.batch_size = batch_size
    self.n_batches = int(math.ceil(X.shape[0] / batch_size))
    self._current = 0 
        
  def __iter__(self):
    return self
    
  def __next__(self):
    return self.next()
    
  def next(self):
    if self._current >= self.n_batches:
      raise StopIteration()
    k = self._current
    self._current += 1
    bs = self.batch_size
    X_batch = torch.LongTensor(self.X[k*bs:(k + 1)*bs])
    Y_batch = torch.FloatTensor(self.Y[k*bs:(k + 1)*bs])

    return X_batch, Y_batch.view(-1, 1)