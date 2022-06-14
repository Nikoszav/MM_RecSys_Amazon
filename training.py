# import torch
# import config
# from training import train_vqa
# from model import get_IQA, get_language_encoder
# from datasets import get_dataset

# #
# # Train VQA
# #
# train_dataset, val_dataset = get_dataset(
#    data_source = "IQA",
#    dataset_folder = config.coco_folder_location,
#    annotation_file = config.vqa_annotation_file_location
#    )


# model = get_IQA()

# optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# best_val_loss=1000
# if config.resume_checkpoint:
#    save_path = f"checkpoints/model_{config.check_point_load}.pth"
#    checkpoint_data = torch.load(save_path)
#    best_val_loss = checkpoint_data["best_loss"]
#    model.load_state_dict(checkpoint_data['state_dict'])
#    print("=> loaded checkpoint '{}' (epoch {}). Best valditation loss: {}"
#          .format(config.check_point_load, checkpoint_data['epoch'], best_val_loss))   

# if torch.cuda.device_count() > 1 and config.use_gpu:
#    print(f"Using multi-gpu: Devices={config.number_devices}")
#    config.device = torch.cuda.current_device()
#    model.to(config.device)
#    model = torch.nn.DataParallel(module=model, device_ids = [i for i in range(torch.cuda.device_count())]).cuda()
# else:
#    model.to(config.device)

# # This needs to be done after moving the model to the gpu 
# if config.resume_checkpoint:
#    optimizer.load_state_dict(checkpoint_data['optimizer'])
#    print("=> loaded optimizer")   

# train_vqa(model, optimizer, train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=100, best_val_loss=best_val_loss)



# We define an iterator to go over the dataset in batches

import numpy as np
import math
import torch
import torch as nn
# from model import NeuralColabFilteringNet
from datasets import DatasetBatchIterator

def train(ncf, datasets):

   # Reset model's parameters, in case the cell is re-run
   ncf._init_params()
   ncf.train()

   import time
   from torch import optim
   import copy

   # Hyper parameters
   lr = 1e-5
   wd = 1e-4
   batch_size = 2560
   max_epochs = 50
   early_stop_epoch_threshold = 3

   # Training loop control parameters
   no_loss_reduction_epoch_counter = 0
   min_loss = np.inf
   min_loss_model_weights = None
   history = []
   iterations_per_epoch = int(math.ceil(len(datasets['train']) // batch_size))
   min_epoch_number = 1
   epoch_start_time = 0

   # Setup GPU, if available, else default to CPU
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   ncf.to(device)
   print(f'Device configured: {device}')

   # Configure loss and optimizer
   loss_criterion = torch.nn.MSELoss(reduction='sum')
   optimizer = optim.Adam(ncf.parameters(), lr=lr, weight_decay=wd)

   # Training loop - iterate over epochs, with early stopping
   print(f'Starting training loop...')
   training_start_time = time.perf_counter()
   
   
   
   for epoch in range(max_epochs):
      stats = {'epoch': epoch + 1, 'total': max_epochs}
      epoch_start_time = time.perf_counter()
      
      for phase in ('train', 'test'):
         is_training = phase == 'train'
         ncf.train(is_training)
         running_loss = 0.0
         n_batches = 0
         
         # Iterate on train/test datasets in batches
         for x_batch, y_batch in DatasetBatchIterator(datasets[phase][0], datasets[phase][1], batch_size=batch_size, shuffle=is_training):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_training):
               outputs = ncf(x_batch[:, 0], x_batch[:, 1])
               # print(x_batch[0,1])
               loss = loss_criterion(outputs, y_batch)
               
               if is_training:
                  loss.backward()
                  optimizer.step()
            
            running_loss += loss.item()
            
         epoch_loss = running_loss / len(datasets[phase][0])
         stats[phase] = epoch_loss
         history.append(stats)
         
         if phase == 'test':
            stats['time'] = time.perf_counter() - epoch_start_time
            print('Epoch [{epoch:03d}/{total:03d}][Time:{time:.2f} sec] Train Loss: {train:.4f} / Validation Loss: {test:.4f}'.format(**stats))
            if epoch_loss < min_loss:
               min_loss = epoch_loss
               min_loss_model_weights = copy.deepcopy(ncf.state_dict())
               no_loss_reduction_epoch_counter = 0
               min_epoch_number = epoch + 1
            else:
               no_loss_reduction_epoch_counter += 1 
   
      if no_loss_reduction_epoch_counter >= early_stop_epoch_threshold:
         print(f'Early stopping applied. Minimal epoch: {min_epoch_number}')
         break     
   print(f'Training completion duration: {(time.perf_counter() - training_start_time):.2f} sec. Validation Loss: {min_loss}')      
      