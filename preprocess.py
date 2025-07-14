from dataloading import create_dataset
from main import CONFIG

folder = CONFIG["PATH"]+"/datasets"

for data_type in ["train", "val", "test"]:
    print(f"Creating {data_type}...")
    train_dataset = create_dataset(data_type, CONFIG, verbose=True)
    print(f"Saving {data_type}...")
    train_dataset.save(f"{folder}/{data_type}.pt")