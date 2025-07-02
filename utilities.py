from collections import defaultdict as dd
import pandas as pd
from tqdm import tqdm
import os
import json
import gzip
from tqdm import tqdm

RUN_DATA_PATH = "./run_data/"

def process_zips():
    for filename in tqdm(os.listdir(RUN_DATA_PATH), desc="Unzipping files"):
        full_name = RUN_DATA_PATH+filename

        if filename[-2:] == "gz":
            # unzip file
            with gzip.open(full_name, "rb") as old_file:
                new_name = full_name[:-3]
                with open(new_name, "wb") as new_file:
                    new_file.write(old_file.read())
            
            # delete zipped file
            os.remove(full_name)

def pad(thing, coef, columns):
    max_length = max(map(len, columns))
    remaining = max_length + 3 - len(thing) + int(coef >= 0)
    return thing + " "*remaining

def extract_runs(silent=False):
    runs = []

    # load files
    for file_name in tqdm(os.listdir(RUN_DATA_PATH), disable=silent, desc="Extracting runs"):
        if file_name[0] == ".":
            continue

        with open(f"{RUN_DATA_PATH}/{file_name}") as file:
            data = json.load(file)

            for game in data:
                runs.append(game["event"])
    
    return runs

def runs_to_df(runs, threshold=0.01):
    # extract relevant info
    data = []

    for run in runs:
        info = dd(int)

        info["victory"] = run["victory"]

        for relic in run["relics"]:
            info[relic] = 1

        for card in run["master_deck"]:
            info[card] += 1

        info[run["character_chosen"]] = 1

        data.append(info)

    # turn into dataframe
    data = pd.DataFrame.from_dict(data).fillna(0)

    # filter out rare cards
    removing = []

    for column, value in tqdm(data.mean().items()):
        if value < threshold:
            removing.append(column)
        
    data = data.drop(columns=removing)

    return data