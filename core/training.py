import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import wandb
from utilities import *
from dataloading import *
from tqdm.auto import tqdm
from model_code import *
from accelerate import Accelerator, load_checkpoint_and_dispatch
import os

def train(config):
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Logging in...")
        wandb.login()

        # this is the training script, assumes you're using the transformer
        # if you're using the MLP, you'll need to change the data pipeline and the final dimension
        # also you can modify the transformer config in the transformer.py file

        # load the data
        print("Loading data...")

    train_dataloader = get_dataset_and_loader("train", config, verbose=accelerator.is_local_main_process)[1]
    val_dataloader = get_dataset_and_loader("val", config, verbose=accelerator.is_local_main_process)[1]

    # setup the model
    ModelType = MODELS[config["model_type"]]

    model = ModelType(config)

    # optionally: load the model
    path = config["PATH"]
    modelname = config["modelname"]
    save_directory = f"{path}/model/{modelname}"
    file_path = f"{save_directory}/model.safetensors"
    
    if os.path.isfile(file_path):
        model = load_checkpoint_and_dispatch(model, file_path)

    # Define the loss function
    criterion = model.get_loss()

    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    threshold = config["threshold"]

    # Define the optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=lr_patience,
        threshold=threshold
    )

    # set up accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    val_dataloader = accelerator.prepare(val_dataloader)

    if accelerator.is_local_main_process:
        print("Training...")

        # train the model
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="slay-the-spire",

            # track run hyperparameters and metadata
            config=config,
            settings=wandb.Settings(),
            resume="allow",
            id=modelname
        )

    # patience = 45
    # cur_patience = 0
    # best_loss = float("inf")

    epoch = 0

    last_train_loss = None
    last_val_loss = None

    # training loop
    while True:
        epoch += 1
        model.train()  # Set the model to training mode

        total_train_loss = 0.0
        total_train_accuracy = 0.0
        num_batches = 0

        if accelerator.is_local_main_process:
            print("Training...")
        
        for batch in tqdm(train_dataloader, disable=not accelerator.is_local_main_process):
            # get input and targets
            model_input, targets = batch[:-1], batch[-1]
            targets = targets.reshape(-1) # fix tensor shape (specific to this project)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(*model_input)  # Forward pass

            loss = criterion(outputs, targets)  # Calculate the loss
            accelerator.backward(loss)  # Backward pass
            optimizer.step()  # Update weights

            # stat track
            total_train_loss += loss.item()
            accuracy = calculate_accuracy(outputs, targets)
            total_train_accuracy += accuracy
            num_batches += 1

        average_train_accuracy = total_train_accuracy / num_batches
        train_loss = total_train_loss / num_batches

        # Calculate and print accuracy after each epoch
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode

            # calculate validation stats
            total_accuracy = 0.0
            total_loss = 0.0

            num_batches = 0

            if accelerator.is_local_main_process:
                print("Evaluating...")

            for batch in tqdm(val_dataloader, disable=not accelerator.is_local_main_process):
                # get input and targets
                model_input, targets = batch[:-1], batch[-1]

                targets = targets.reshape(-1) # fix tensor shape (specific to this project)

                outputs = model(*model_input)

                all_outputs, all_targets = accelerator.gather_for_metrics((outputs, targets))

                # calculate the val accuracy
                accuracy = calculate_accuracy(outputs, targets)
                total_accuracy += accuracy

                # Calculate the val loss
                loss = criterion(all_outputs, all_targets)
                total_loss += loss.item()
                num_batches += 1

            average_accuracy = total_accuracy / num_batches
            val_loss = total_loss / num_batches

            metrics = {
                "validation_accuracy": average_accuracy,
                "loss": val_loss,
                "training_accuracy": average_train_accuracy,
                "training_loss": train_loss
            }

            # to show how fast we're plateauing
            if epoch > 1:
                metrics["delta_train_loss"] = train_loss - last_train_loss
                metrics["delta_val_loss"] = val_loss - last_val_loss
            
            last_train_loss = train_loss
            last_val_loss = val_loss

            if accelerator.is_local_main_process:
                print(f"Epoch {epoch + 1}, Train loss {train_loss} Train Accuracy {average_train_accuracy} Validation Accuracy: {average_accuracy}, Val loss: {val_loss}")

                # log metrics to wandb
                wandb.log(metrics)
            
        # always save the model
        accelerator.wait_for_everyone()
        accelerator.save_model(model, f"{save_directory}")
        
        # save embedding pictures so we can make gifs later
        # this is broken since we added accelerate
        # TODO: FIX this later
        # if accelerator.is_local_main_process:
        #     save_embedding_pictures(model)

        # learning rate scheduling
        scheduler.step(train_loss)