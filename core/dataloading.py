from utilities import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32
from state_analysis import extract_states_and_choices, extract_states
import torch
import os

MAX_OPTIONS_LENGTH = 5

class SimpleDataset(Dataset):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return len(self.state_cat)

    def __getitem__(self, index):
        sample = (
            self.state_cat[index],
            self.state_cont[index],
            self.targets[index]
        )

        return sample
    
    def save(self, location):
        torch.save(self, location)

    def append(self, states, verbose=False):
        state_cat, state_cont, targets = self.process_data(states, verbose=verbose)

        new_state_cat = tensor(state_cat, dtype=int)
        new_state_cont = tensor(state_cont, dtype=float32)
        new_targets = tensor(targets, dtype=int)

        self.state_cat = torch.cat((self.state_cat, new_state_cat))
        self.state_cont = torch.cat((self.state_cont, new_state_cont))
        self.targets = torch.cat((self.targets, new_targets))

class SkipBotDataset(SimpleDataset):
    def __init__(self, states, config, choices=None, verbose=False, *args, **kwargs):
        super().__init__(config)

        state_cat, state_cont, card_choices, targets = self.process_data(states, choices, verbose=verbose)
      
        self.state_cat = tensor(state_cat, dtype=int)
        self.state_cont = tensor(state_cont, dtype=float32)
        self.targets = tensor(targets, dtype=int)
        self.card_choices = tensor(card_choices, dtype=float32)
        
    def __getitem__(self, index):
        sample = (
            self.state_cat[index],
            self.state_cont[index],
            self.card_choices[index],
            self.targets[index]
        )

        return sample
    
    def append(self, states, choices, verbose=False):
        state_cat, state_cont, card_choices, targets = self.process_data(states, choices, verbose)

        new_state_cat = tensor(state_cat, dtype=int)
        new_state_cont = tensor(state_cont, dtype=float32)
        new_card_choices = tensor(card_choices, dtype=int)
        new_targets = tensor(targets, dtype=int)

        self.state_cat = torch.cat((self.state_cat, new_state_cat))
        self.state_cont = torch.cat((self.state_cont, new_state_cont))
        self.card_choices = torch.cat((self.card_choices, new_card_choices))
        self.targets = torch.cat((self.targets, new_targets))
    
    def process_data(self, states, choices, verbose=False, *args):
        state_cat = []
        state_cont = []
        card_choices = []
        targets = []

        discard_too_long = self.config["discard_too_long"]
        max_cat_length = self.config["max_cat_length"]

        for state, choice in tqdm(
            zip(states, choices), 
            desc="Processing s&c...", 
            disable=not verbose,
            total=len(states)
        ):
            # get categorical data
            # order: character, deck, relics, choices
            cat_data = [state["character"]] + state["deck"] + state["relics"]

            # skip very long runs
            if len(cat_data) > max_cat_length and discard_too_long:
                if verbose:
                    print("Too long, skipping...")
                continue
            
            cat_data = pad_cat_data(cat_data, max_cat_length)

            tok_state = tokenize_list(cat_data)

            # get cont data
            # order: alphabetical
            cont_data = [
                state["ascension"],
                state["current_hp"],
                state["floor"],
                state["gold"],
                state["max_hp"],
                state["victory"]
            ]
            
            # check for modded runs
            if len(choice["options"]) > MAX_OPTIONS_LENGTH:
                if verbose:
                    print("Weirdly long options, skipping")
                continue

            # get card choices
            options = pad_cat_data(choice["options"], MAX_OPTIONS_LENGTH)

            tok_cards = tokenize_list(options, category="cards")

            # skip modded runs
            missing_token = tokenize(MISSING)
            if missing_token in tok_state or missing_token in tok_cards:
                if verbose:
                    print("Missing token, skipping...")
                continue

            state_cat.append(tok_state)
            state_cont.append(cont_data)
            card_choices.append(tok_cards)

            # get target
            targets.append(tokenize(choice["picked"], category="cards"))
        
        return state_cat, state_cont, card_choices, targets

class V2Dataset(SimpleDataset):
    def __init__(self, states, config, verbose=False, *args, **kwargs):
        super().__init__(config)

        state_cat, state_cont, targets = self.process_data(states, verbose=verbose)
      
        self.state_cat = tensor(state_cat, dtype=int)
        self.state_cont = tensor(state_cont, dtype=float32)
        self.targets = tensor(targets, dtype=float32)

    def process_data(self, states, verbose=False):
        state_cat = []
        state_cont = []
        targets = []

        discard_too_long = self.config["discard_too_long"]
        max_cat_length = self.config["max_cat_length"]

        for state in tqdm(
            states, 
            desc="Processing s&c...", 
            disable=not verbose,
            total=len(states)
        ):
            # get categorical data
            # order: character, deck, relics, choices
            cat_data = [state["character"]] + state["deck"] + state["relics"]

            # skip very long runs
            if len(cat_data) > max_cat_length and discard_too_long:
                if verbose:
                    print("Too long, skipping...")
                continue
            
            cat_data = pad_cat_data(cat_data, max_cat_length)

            tok_state = tokenize_list(cat_data)

            # get cont data
            # order: alphabetical
            cont_data = [
                state["ascension"],
                state["current_hp"],
                state["deck_size"],
                state["floor"],
                state["floors_to_boss"],
                state["gold"],
                state["max_hp"],
                state["n_attacks"],
                state["n_curses"],
                state["n_powers"],
                state["n_relics"],
                state["n_skills"]
            ]

            # skip modded runs
            missing_token = tokenize(MISSING)
            if missing_token in tok_state:
                if verbose:
                    print("Missing token, skipping...")
                continue

            state_cat.append(tok_state)
            state_cont.append(cont_data)

            # get target
            targets.append(state["victory"])
        
        return state_cat, state_cont, targets

DATASETS = {
    "skip-bot": SkipBotDataset,
    "v2": V2Dataset
}

def create_dataset(data_type, config, verbose=False):
    DataSetType = DATASETS[config["model_type"]]

    should_speak = verbose

    # do train states
    if should_speak:
        print(f"Loading {data_type} data...")
        
    path = f"./run_data/{data_type}"
    process_zips(path, verbose=should_speak)
    
    # load data in batches
    # batch size may be really large
    batchsize = config["file_batchsize"]
    filenames = os.listdir(path)

    dataset = None

    for batch in tqdm(batched(filenames, batchsize), disable=not should_speak, desc="Processing batches..."):
        runs = extract_runs(path, files=batch, verbose=should_speak)

        if config["model_type"] == "skip-bot":
            states, choices = extract_states_and_choices(runs, verbose=should_speak)
        elif config["model_type"] == "v2":
            states = extract_states(runs, verbose=should_speak)
            choices = []
        else:
            raise Exception("Unknown model type")

        if dataset is None:
            dataset = DataSetType(states, config, choices=choices, verbose=should_speak)
        else:
            dataset.append(states, choices, verbose=should_speak)
    
    return dataset

# allow proper loading of preprocessed datasets
torch.serialization.add_safe_globals([
    SkipBotDataset,
    V2Dataset
])

# data type can be train, val, or test
def get_dataset_and_loader(data_type, config, verbose=False):
    # check if we are loading prepocessed data
    # we usually will be
    if config["use_preprocessed_data"]:
        folder = config["PATH"]+"/datasets/"+config["model_type"]
        file_loc = f"{folder}/{data_type}.pt"

        if verbose:
            print(f"Loading {data_type} datatset from file...")

        # check the data exists
        if f"{data_type}.pt" not in os.listdir(folder):
            raise Exception(
                f"{data_type} saved dataset does not exist. Run preprocessing.py to generate."
            )

        # load the dataset
        dataset = torch.load(file_loc)
    else:
        if verbose:
            print(f"Creating {data_type} dataset...")
        dataset = create_dataset(data_type, config, verbose)

    batchsize, n_workers = config["batchsize"], config["n_workers"]
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=n_workers)

    return dataset, dataloader