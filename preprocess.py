from dataloading import create_dataset
from main import CONFIG

FOLDER = CONFIG["PATH"]+"/datasets"

for data_type in ["train", "val", "test"]:
    print(f"Creating {data_type}...")
    dataset = create_dataset(data_type, CONFIG, verbose=True)
    print(f"Saving {data_type}...")
    dataset.save(f"{FOLDER}/{data_type}.pt")