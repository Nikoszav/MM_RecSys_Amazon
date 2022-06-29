import numpy as np
import math
import torch
import torch as nn

# from model import NeuralColabFilteringNet
from datasets.datasets import DatasetBatchIterator

# TODO
# Tutorial for wandb: https://www.youtube.com/watch?v=G7GH0SeNBMA&ab_channel=Weights%26Biases
# Another tutorial for NCF: https://gist.github.com/khuangaf/bf2a216019d29a4a1014f71dbfff51d0  // https://www.youtube.com/watch?v=O4lk9Lw7lS0&ab_channel=SteeveHuang
# See architechure of this tutorial https://medium.com/coinmonks/how-to-implement-a-recommendation-system-with-deep-learning-and-pytorch-2d40476590f9
# Add a sheduler as well for the lr
# Work on DLRM
# See NEUMF and GMF good tutorial: https://d2l.ai/chapter_recommender-systems/neumf.html
# another way "to concat" is by mutiplying (see maybe this: https://github.com/cheon-research/J-NCF-pytorch/blob/master/model.py)
# see negative sampling and other cost functions
# Tutorial for NCF have a look when time: https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb
# Repo from NVIDIA wth a lot of different recomenders: https://github.com/NVIDIA/DeepLearningExamples

# for comments colors


import wandb

from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train(ncf, datasets, hyperparameters):

    # Reset model's parameters, in case the cell is re-run
    ncf._init_params()
    ncf.train()

    import time
    from torch import optim
    import copy

    # Hyper parameters
    # lr = 1e-3
    # wd = 1e-4
    # batch_size = 1020
    # max_epochs = 60
    # early_stop_epoch_threshold = 4
    lr = hyperparameters["lr"]
    wd = hyperparameters["wd"]
    batch_size = hyperparameters["batch_size"]
    max_epochs = hyperparameters["max_epochs"]
    early_stop_epoch_threshold = hyperparameters["early_stop_epoch_threshold"]

    # Training loop control parameters
    no_loss_reduction_epoch_counter = 0
    min_loss = np.inf
    min_loss_model_weights = None
    history = []
    iterations_per_epoch = int(math.ceil(len(datasets["train"]) // batch_size))
    min_epoch_number = 1
    epoch_start_time = 0

    # Setup GPU, if available, else default to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    ncf.to(device)
    print(f"Device configured: {device}")
    print(next(ncf.parameters()).is_cuda)

    # Configure loss and optimizer
    # loss_criterion = torch.nn.MSELoss(reduction='sum')
    loss_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    # loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion = loss_criterion.to(device)
    optimizer = optim.Adam(ncf.parameters(), lr=lr, weight_decay=wd)

    # Training loop - iterate over epochs, with early stopping
    print(f"Starting training loop...")
    training_start_time = time.perf_counter()
    wandb.watch(ncf, criterion=loss_criterion, log="all", log_freq=10)

    for epoch in tqdm(range(max_epochs)):
        stats = {"epoch": epoch + 1, "total": max_epochs}
        stats_acc = {"epoch": epoch + 1, "total": max_epochs}
        epoch_start_time = time.perf_counter()

        for phase in ("train", "test"):
            is_training = phase == "train"
            ncf.train(
                is_training
            )  # it says to the model that you are start the training
            running_loss = 0.0
            n_batches = 0  # count the number of batcg
            running_acc = 0

            # Iterate on train/test datasets in batches
            for x_batch, y_batch in DatasetBatchIterator(
                datasets[phase][0],
                datasets[phase][1],
                batch_size=batch_size,
                shuffle=False,
            ):
                n_batches += 1
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # print(x_batch[0])
                # print(y_batch[0])
                optimizer.zero_grad()
                groud_truth = []
                predictions = []
                with torch.set_grad_enabled(
                    is_training
                ):  # you set the calculation of gradients to be on

                    outputs = ncf(x_batch[:, 0], x_batch[:, 32], x_batch[:, 1:32])
                    # print(outputs[0])
                    # outputs = outputs.softmax(dim=1)
                    # print(outputs.size())
                    # print(y_batch.size())
                    loss = loss_criterion(outputs, y_batch)
                    # print(outputs[0])
                    # print(y_batch[0])
                    # print(outputs)
                    act_label = torch.argmax(
                        outputs, dim=1
                    ).cpu()  # act_label = 1 (index)
                    pred_label = torch.argmax(
                        y_batch, dim=1
                    ).cpu()  # act_label = 1 (index)
                    groud_truth.extend(act_label)
                    predictions.extend(pred_label)
                    # pred_label = np.argmax(outputs.cpu().detach().numpy()) # pred_label = 1 (index)
                    # print(act_label.cpu())
                    # print(pred_label.cpu())
                    # acc =
                    # print(acc)
                    # acc = accuracy_score(y_batch.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    # print(y_batch[0])

                    # a = torch.nn.functional.one_hot(y_batch.to(torch.int64), num_classes=5)
                    # print(a.shape)
                    # groud_truth.extend(y_batch.tolist())
                    # predictions.extend(outputs.tolist())

                    if is_training:
                        loss.backward()
                        optimizer.step()

                # running_acc =
                running_loss += loss.item()

            epoch_loss = running_loss / len(datasets[phase][0])
            epoch_acc = accuracy_score(groud_truth, predictions)

            wandb.log(
                {"epoch": epoch, "loss": epoch_loss, "loss": epoch_acc}
            )  # maybe you need to pass something there for every step you need the report
            stats[phase] = epoch_loss
            stats_acc[phase] = epoch_acc
            history.append(stats)

            if phase == "test":
                stats["time"] = time.perf_counter() - epoch_start_time
                stats_acc["time"] = time.perf_counter() - epoch_start_time
                print(
                    "Epoch [{epoch:03d}/{total:03d}][Time:{time:.2f} sec] Train Loss: {train:.4f} / Validation Loss: {test:.4f}".format(
                        **stats
                    )
                )
                print(
                    "Epoch [{epoch:03d}/{total:03d}][Time:{time:.2f} sec] Train Acc: {train:.4f} / Validation Acc: {test:.4f}".format(
                        **stats_acc
                    )
                )

                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    min_loss_model_weights = copy.deepcopy(
                        ncf.state_dict()
                    )  # TODO: When you finalize the model save this to a forlder
                    no_loss_reduction_epoch_counter = 0
                    min_epoch_number = epoch + 1
                else:
                    no_loss_reduction_epoch_counter += 1

        if no_loss_reduction_epoch_counter >= early_stop_epoch_threshold:
            print(f"Early stopping applied. Minimal epoch: {min_epoch_number}")
            break
    print(
        f"Training completion duration: {(time.perf_counter() - training_start_time):.2f} sec. Validation Loss: {min_loss}"
    )
