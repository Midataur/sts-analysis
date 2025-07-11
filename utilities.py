from collections import defaultdict as dd
from game_data import VOCABULARY, CARDS_LIST
from torch import argmax
import pandas as pd
from tqdm import tqdm
import os
import json
import gzip

def process_zips(data_path):
    for filename in tqdm(os.listdir(data_path), desc="Unzipping files"):
        full_name = data_path+filename

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

def format_choice(choice):
    options = choice["not_picked"]

    # SKIP is always implicitly an option
    # no need to add it explicitly
    if choice["picked"] != "SKIP":
        options.append(choice["picked"])

    return {
        "options": options,
        "picked": choice["picked"]
    }

def extract_runs(data_path, silent=False):
    runs = []

    # load files
    for file_name in tqdm(os.listdir(data_path), disable=silent, desc="Extracting runs"):
        if file_name[-5:] != ".json":
            continue

        with open(f"{data_path}/{file_name}") as file:
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

def tokenize(item, category=None):
    # return special category token
    if category == "cards":
        return CARDS_LIST.index(item)
    
    # return regular token
    return VOCABULARY.index(item)

def tokenize_list(cat_data):
    return list(map(tokenize, cat_data))

def calculate_accuracy(output, target):
    # targets is a (B) tensor of integers that have the index of the correct class
    # we need to see if the max logit is at the right index
    return (argmax(output, dim=1) == target).float().mean()