print("Loading libraries...")

from training import train

CONFIG = {
    "n_embed": 402,
    "n_heads": 6,
    "dropout": 0,
    "n_blocks": 4,
    "context_length": None,
    "n_cont": 6,
    "modelname": "starter-1",
    "learning_rate": 3*(10**-5),
    "batchsize": 64,
    "weight_decay": 0.05,
    "lr_factor": 0.1,
    "lr_patience": 10,
    "threshold": 0.01,
    "n_workers": 0,
    "max_cat_length": 120, #decided emperically, never needed more than 100
    "PATH": ".",
}

assert CONFIG["n_embed"] % CONFIG["n_heads"] == 0

if __name__ == "__main__":
    train(CONFIG)