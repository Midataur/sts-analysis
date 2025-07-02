from sklearn.linear_model import LogisticRegression
from collections import defaultdict as dd
from tqdm import tqdm
from utilities import *
import json
import pandas as pd
import os

runs = extract_runs()

data = runs_to_df(threshold=0.015)

# run a logistic regression
X, y = data.drop(columns="victory"), data["victory"]

model = LogisticRegression(penalty="l1", solver="liblinear").fit(X, y)

for thing, coef in sorted(zip(X.columns, model.coef_.flatten()), key=lambda x: -x[1]):
    print(pad(thing, coef, X.columns), "%.4f"%coef)