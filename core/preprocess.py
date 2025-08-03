from dataloading import create_dataset
import multiprocessing as mp
from main import CONFIG
import torch
import sys

FOLDER = CONFIG["PATH"]+"/datasets/"+CONFIG["model_type"]

if __name__ == "__main__":
    # ensures consistency across machines
    mp.set_start_method("spawn")

    # Configure output for immediate flushing
    sys.stdout.reconfigure(line_buffering=True)

    torch.set_default_device("cpu")

    for data_type in ["train", "val", "test"]:
        print(f"Creating {data_type}...")
        dataset = create_dataset(data_type, CONFIG, verbose=True)
        print(f"Saving {data_type}...")
        dataset.save(f"{FOLDER}/{data_type}.pt")
        print(f"\nSaved {data_type}\n")