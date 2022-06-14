# import torch
# from tqdm import tqdm
# import config
# import time
# import torch.nn as nn
# import torch.optim as optim
	
# criterion = torch.nn.CrossEntropyLoss()

# def train_vqa(model, optimizer, train_dataset, val_dataset=None, num_epochs=50, best_val_loss=1000):	
# 	scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

# 	for epoch in range(num_epochs):
# 		model.train()

# 		epoch_running_loss = 0

# 		total_samples = 0
# 		correct_samples = 0

# 		pbar = tqdm(train_dataset)
# 		b = 0

# 		for batch in pbar:
# 			b+=1
# 			if config.number_devices > 1:
# 				batch = [t.squeeze() for t in batch]
# 			else:
# 				batch = [t.squeeze().to(config.device) for t in batch]
			
# 			try:
# 				ground_truths = batch[-1].cuda(non_blocking=True)
				
# 			except:
# 				ground_truths = batch[-1]

# 			i, q, q_mask, a, a_mask = batch[:-1]
# 			predictions = model(i, q, q_mask, a, a_mask)
			
# 			loss = criterion(predictions, ground_truths)
# 			correct_samples += calculate_correct(predictions, ground_truths).item()
# 			total_samples += ground_truths.shape[0]
			
# 			loss.backward()
# 			# torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
# 			optimizer.step()
# 			scheduler.step()

# 			epoch_running_loss += loss.item()
# 			model.zero_grad()

# 			pbar.set_description("Epoch %s: train loss %s, train acc %s" % (epoch, round(epoch_running_loss/b, 5), round(correct_samples * 100 / total_samples, 3)))

# 		# val_acc, val_loss = evaluate(model, val_dataset)
# 		# print("Epoch %s: val loss %s, val acc %s" % (epoch, round(val_loss, 5), round(val_acc, 3)))

# 		# if val_loss < best_val_loss:
# 		# 	save_path = f"checkpoints/model_{config.check_point_name}.pth"
# 		# 	print(f"New best model saved to {save_path}. Val loss: {best_val_loss}")
# 		# 	best_val_loss = val_loss
# 		# 	checkpoint_data = {
#         #         'epoch': epoch,
#         #         'state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
#         #         'best_loss': best_val_loss,
#         #         'optimizer' : optimizer.state_dict(),
#         #     }
# 		# 	torch.save(checkpoint_data, save_path)

# def evaluate(model, dataset):
# 	model.eval()
	
# 	pbar = tqdm(dataset)
# 	pbar.set_description("Evaluating")

# 	total_samples = 0
# 	correct_samples = 0
# 	total_loss = 0

# 	with torch.no_grad():

# 		for batch in pbar:
# 			batch = [t.squeeze().to(config.device) for t in batch]
# 			ground_truths = batch[-1].to(config.device)
# 			predictions = model(*batch[:-1])

# 			total_loss += criterion(predictions, ground_truths).item()
# 			correct_samples += calculate_correct(predictions, ground_truths).item()
# 			total_samples += ground_truths.shape[0]

# 		accuracy = correct_samples * 100 / total_samples
# 		loss = total_loss / total_samples

# 		return accuracy, loss


# def calculate_correct(predictions, ground_truths):
# 	p = torch.argmax(predictions, dim=1)
# 	return torch.sum(p==ground_truths)


# def gif_preproc(model, dataset, save_folder):
# 	import pickle
# 	model.to(config.device)

# 	for batch in tqdm(dataset):
# 		filenames, gifs, masks = batch

# 		# Pass images through model
# 		with torch.no_grad():
# 			feature_tensor = model(gifs.to(config.device)).cpu()
		
# 		# For sample in batch
# 		for i in range(config.batch_size):
# 			filename_i = filenames[i]
# 			tensor_i = feature_tensor[i]
# 			mask_i = masks[i]

# 			# Store tensor and mask
# 			feature_dict = {'tensor':tensor_i,'mask':mask_i}

# 			# Save feature dict to pickle
# 			file_name = save_folder + '/' + filename_i + '.pkl' 
			
# 			with open(file_name, "wb") as handle:
# 				pickle.dump(feature_dict, handle)

# def text_preproc(lan_model, dataset, save_folder, q_max_length=14, a_max_length=8):
# 	import pickle
# 	lan_model.to(config.device)

# 	for batch in tqdm(dataset):
# 		filenames, questions, answers = batch
		
# 		# Create question and answer tokens
# 		q_tokens, q_masks = lan_model.tokenize_text(questions, max_length=q_max_length, is_multi_list=False)
# 		a_tokens, a_masks = lan_model.tokenize_text(answers, max_length=a_max_length, is_multi_list=True, transpose_list=True)

# 		# Pass models through language encoder
# 		with torch.no_grad():
# 			q_tensors = lan_model(q_tokens.to(config.device), q_masks.to(config.device)).cpu()

# 			# Stack answers: [batch, num_answers, num_tokens] -> [batch x num_answers, num_tokens]
# 			a_stacked_tokens = torch.reshape(a_tokens, shape=(config.batch_size*5, -1))
# 			a_stacked_masks = torch.reshape(a_masks, shape=(config.batch_size*5, -1))
# 			a_stacked_tensors = lan_model(a_stacked_tokens.to(config.device), a_stacked_masks.to(config.device)).cpu()

# 			# Unstack answers: [batch x num_answers, num_tokens] -> [batch, num_answers, num_tokens]
# 			a_tensors = torch.reshape(a_stacked_tensors, shape=(config.batch_size, 5, a_max_length, -1))

# 		# Loop through each sample in batch and save it to pickle
# 		for i in range(config.batch_size):
# 			feature_dict = {}
# 			filename_i = filenames[i]
# 			q_tensor_i = q_tensors[i]
# 			a_tensor_i = a_tensors[i]

# 			# Find mask idx for questions
# 			q_mask_idx = torch.argmin(q_masks[i]).item()
# 			if q_mask_idx == 0: q_mask_idx = q_max_length

# 			# Store question tensor and mask idx
# 			feature_dict = {'question':q_tensor_i, 'question_mask_idx':q_mask_idx}

# 			for j in range(5):
# 				cur_a_tensor = a_tensor_i[j]
				
# 				# Find mask idx for answers
# 				cur_a_mask_idx = torch.argmin(a_masks[i,j]).item()
# 				if cur_a_mask_idx == 0: cur_a_mask_idx = a_max_length

# 				# Store answer tensor and mask idx
# 				feature_dict[f'a{j+1}'] = cur_a_tensor
# 				feature_dict[f'a{j+1}_mask_idx'] = cur_a_mask_idx

# 			# Save to sample to pickle object
# 			file_name = save_folder + '/' + filename_i + '.pkl'

# 			with open(file_name, "wb") as handle:
# 				pickle.dump(feature_dict, handle)

# 		# Just process the first batch for testing purposes, remove later




# Now let's instantiate an NCF model and try a prediction with uninitialized weights
from datasets import Movies
from model import NeuralColabFilteringNet
import config

import numpy as np
import pandas as pd

ratings = pd.read_pickle(config.ratings_file)
data =  pd.read_pickle(config.data_file)

user_count = ratings["userId"].nunique()
movie_count = data["movieId"].nunique()


# Let's peek into the loaded DataFrames
print('Sneak peek into the ratings DataFrame:\n\n', ratings.head(), '\n')
print(f'Number of users: {user_count:,}')
print(f'Number of ratings: {ratings.shape[0]:,}')
print(f'Number of rated movies: {ratings["movieId"].nunique():,}\n')
print(f'Sneak peek into the movies DataFrame:\n\n', data.head(), '\n')
print(f'Number of movies: {movie_count:,} - notice this is higher than the number of rated movies!')

print("##########################################################################################")

#We marge the df so we can take the metadata for the movies
union = pd.merge(ratings, data, on='movieId', how='inner')
union.sort_values(by = ["timestamp"], inplace=True)
# print (union.head())


# Define our input and labels data X,Y
X = union[['userId','movieId', "timestamp"]]

Y = union['rating'].astype(np.float32)


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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, shuffle=False)
datasets = {'train': (X_train, Y_train), 'test': (X_test, Y_test)}

#Checking the timestamps are right
# print(X_train.tail())
# print(X_test.head())

print(f'Training dataset sample size: {len(X_train):,} positive samples ({len(X_train)/len(X)*100:.0f}%)')
print(f'Test dataset sample size: {len(X_test):,} positive samples ({len(X_test)/len(X)*100:.0f}%)')
print(f'Total dataset sample size: {len(X):,} positive samples (100%)')

print("##########################################################################################")

ncf = NeuralColabFilteringNet(user_count, movie_count)
print(f"Our model architecture:\n\n{ncf}\n")

# Let's look at the model size
num_params = sum(p.numel() for p in ncf.parameters())
print(f'Number of model parameters: {num_params:,}, model training size: {num_params*4/(1024**2):.2f} MB')

from training import train

train(ncf, datasets)