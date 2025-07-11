from utilities import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32
from state_analysis import extract_states_and_choices
from accelerate import Accelerator

MAX_OPTIONS_LENGTH = 5

class SimpleDataset(Dataset):
    def __init__(self, states, choices, config, *args, **kwargs):
        state_cat = []
        state_cont = []
        card_choices = []
        targets = []

        max_cat_length = config["max_cat_length"]

        for state, choice in zip(states, choices):
            # get categorical data
            # order: character, deck, relics, choices
            cat_data = [state["character"]] + state["deck"] + state["relics"]
            cat_data = pad_cat_data(cat_data, max_cat_length)

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
            options = pad_cat_data(choice["options"], MAX_OPTIONS_LENGTH)
            card_choices.append(tokenize_list(options, category="cards"))

            # get target
            targets.append(tokenize(choice["picked"], category="cards"))
      
        self.state_cat = tensor(state_cat, dtype=int)
        self.state_cont = tensor(state_cont, dtype=float32)
        self.card_choices = tensor(card_choices, dtype=int)
        self.targets = tensor(targets, dtype=int)

    def __len__(self):
        return len(self.state_cat)

    def __getitem__(self, index):
        sample = (
            self.state_cat[index],
            self.state_cont[index],
            self.card_choices[index],
            self.targets[index]
        )

        return sample

# data type can be train, val, or test
def create_dataset_and_loader(data_type, config, verbose=False):
    run_data_path = "./run_data"

    accelerator = Accelerator()
    should_speak = verbose and accelerator.is_local_main_process

    # do train states
    if should_speak:
        print(f"Loading {data_type} data...")
        
    path = f"{run_data_path}/{data_type}"
    process_zips(path)
    runs = extract_runs(path)
    states, choices = extract_states_and_choices(runs)

    batchsize, n_workers = config["batchsize"], config["n_workers"]

    # create the dataloaders
    dataset = SimpleDataset(states, choices, config, mainthread=should_speak)
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=n_workers)

    return dataset, dataloader