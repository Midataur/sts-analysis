from collections import defaultdict as dd
from game_data import *
from torch import argmax
from itertools import islice
import pandas as pd
from tqdm import tqdm
import os
import json
import gzip

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def process_zips(data_path, verbose=True):
    for filename in tqdm(os.listdir(data_path), desc="Unzipping files", disable=not verbose):
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

# files can be specified manually, or just a path can be given
def extract_runs(data_path, files=None, verbose=True):
    runs = []

    if files is None:
        files = os.listdir(data_path)

    # load files
    for file_name in tqdm(files, disable=not verbose, desc="Extracting runs"):
        if file_name[-5:] != ".json":
            continue

        with open(f"{data_path}/{file_name}") as file:
            try:
                data = json.load(file)
            except Exception as e:
                print(f"While reading {data_path}/{file_name} the following error was encountered:")
                print(e)
                print("Skipping for now")

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
    if item not in VOCABULARY:
        return VOCABULARY.index(MISSING_TOKEN)

    # return special category token
    if category == "cards":
        return AUGMENTED_CARDS_LIST.index(item)
    
    # return regular token
    return VOCABULARY.index(item)

def tokenize_list(cat_data, category=None):
    return [tokenize(x, category) for x in cat_data]

def calculate_accuracy(output, target):
    # targets is a (B) tensor of integers that have the index of the correct class
    # we need to see if the max logit is at the right index
    return (argmax(output, dim=1) == target).float().mean()

def pad_cat_data(cat_data, max_cat_length):
    if len(cat_data) > max_cat_length:
        msg = f"cat_data is longer than max_cat_length ({len(cat_data)} > {max_cat_length})."
        raise Exception(msg)

    remaining = max_cat_length - len(cat_data)
    return cat_data + [EMPTY_TOKEN]*remaining