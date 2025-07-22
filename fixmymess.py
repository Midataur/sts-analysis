from game_data import MISSING
from dataloading import get_dataset_and_loader
from main import CONFIG
from utilities import *

# loads a dataset, replaces malformed tokens, and saves it again
new_cf = dict(CONFIG)
new_cf["use_preprocessed_data"] = True

FOLDER = CONFIG["PATH"]+"/datasets"

MALFORMED = tokenize(MISSING)
CORRECT = tokenize(MISSING, category="cards")

for data_type in ["train", "val", "test"]:
    # load the dataset
    dataset, _ = get_dataset_and_loader(data_type, new_cf, verbose=True)

    # replace the malformed tokens
    print("Fixing...")
    dataset.state_cat[dataset.state_cat == MALFORMED] = CORRECT

    print("Saving...")
    # save the dataset
    dataset.save(f"{FOLDER}/{data_type}.pt")
