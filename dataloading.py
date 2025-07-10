from utilities import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32, cuda, device
import numpy as np
from scipy import sparse
from tqdm import tqdm
from accelerate import Accelerator
import os

class SimpleDataset(Dataset):
    def __init__(self, states, choices, *args, **kwargs):
        state_cat = []
        state_cont = []
        card_choices = []
        targets = []

        for state, choice in zip(states, choices):
            # get categorical data
            # order: character, deck, relics, choices
            cat_data = state["character"] + state["deck"] + state["relics"]
            state_cat.append(tokenize_list(cat_data))

            # get cont data
            # order: alphabetical
            state_cont.append([
                state["ascension"],
                state["current_hp"],
                state["floor"],
                state["gold"],
                state["max_hp"],
                state["victory"]
            ])

            # get card choices
            card_choices.append(tokenize_list(choice["options"], category="cards"))

            # get target
            targets.append(tokenize(choice["picked"], category="cards"))
      
        self.state_cat = tensor(state_cat, dtype=int)
        self.state_cont = tensor(state_cat, dtype=float)
        self.card_choices = tensor(card_choices, int)
        self.targets = tensor(targets, dtype=int)

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, index):
        sample = (
            self.state_cat[index],
            self.state_cont[index],
            self.card_choices[index],
            self.targets[index]
        )

        return sample
        
def load_data(skip_train=False, verbose=False):
    run_data_path = "./run_data"

    accelerator = Accelerator()
    should_speak = verbose and accelerator.is_local_main_process

    train_inputs = np.array([[0 for x in range(INPUT_LENGTH)]])
    train_perms = np.array([[0 for x in range(MAX_GROUP_SIZE)]])

    if not skip_train:
        if should_speak:
            print("Loading training data...")
        
        
    else:
        train_inputs = None
        train_perms = None
        dataset_size = None

    if should_speak:
         print("Loading validation data...")

    val_seqs = np.loadtxt(PATH + DATA + "val_data.csv", delimiter=",").astype(int)
    val_perms = np.loadtxt(PATH + DATA + "val_data_perms.csv", delimiter=",").astype(int)

    if should_speak:
         print("Loading test data...")

    test_seqs = np.loadtxt(PATH + DATA + "test_data.csv", delimiter=",").astype(int)
    test_perms = np.loadtxt(PATH + DATA + "test_data_perms.csv", delimiter=",").astype(int)

    # create the dataloaders
    if not skip_train:
        train_dataset = dataset_class(train_inputs, train_perms, question=question, mainthread=should_speak)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)
    else:
        train_dataset = None
        train_dataloader = None

    val_dataset = dataset_class(val_seqs, val_perms, question=question, mainthread=should_speak)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)

    test_dataset = dataset_class(test_seqs, test_perms, question=question, mainthread=should_speak)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)

    return (
         train_inputs, train_perms, train_dataloader, 
         val_seqs, val_perms, val_dataloader,
         test_seqs, test_perms, test_dataloader,
         dataset_size
    )