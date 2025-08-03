from dataloading import create_dataset
import multiprocessing as mp
from main import CONFIG

FOLDER = CONFIG["PATH"]+"/datasets/"+CONFIG["model_type"]

if __name__ == "__main__":
    # ensures consistency across machines
    mp.set_start_method("spawn")

    for data_type in ["train", "val", "test"]:
        print(f"Creating {data_type}...")
        dataset = create_dataset(data_type, CONFIG, verbose=True)
        print(f"Saving {data_type}...")
        dataset.save(f"{FOLDER}/{data_type}.pt")