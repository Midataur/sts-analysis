CONFIG = {
    "modelname": "ipray-2",
    "model_type": "v2",

    # embedding dimension
    # good starting value: 402
    "n_embed": 402,

    # number of attention heads:
    # good starting value: 6
    # n_embed % n_heads must be 0
    "n_heads": 6,

    # dropout factor to use
    # i usually set to zero
    "dropout": 0,

    # number of blocks to have
    # higher means a deeper network
    "n_blocks": 6,

    # number of continuous variables
    # currently 12, may change
    "n_cont": 12,

    # good starting value: 3*10^-4
    "learning_rate": 3*(10**-4), 

    # good starting value: 64
    "batchsize": 512, 

    # good starting value: 0.1
    "weight_decay": 0.1, 

    # usually 0.1
    "lr_factor": 0.1, 

    # usually 10
    "lr_patience": 10, 

    # usually 0.01
    "threshold": 0.01, 

    # number of workers to use for loading data to the gpus
    # set to 0 for all, +ve for specific
    # usually 0
    "n_workers": 0, 

    # maximum number of categorical variables
    # decided empirically, never really need more than 200
    "max_cat_length": 200, 

    # when loading data, what do we do if there are too many categorical variables?
    # if true, display a warning and skip that one
    # if false, throw an error
    # usually want true, since outliers aren't worth an error
    "discard_too_long": True, # set to true unless you want to fight your data

    # should be . unless you're doing something weird
    "PATH": ".",

    # should we load in preprocessed data?
    # if true, does that
    # if false, processes the data fresh
    # true assumes you have actually run preprocess.py
    "use_preprocessed_data": True
}

assert CONFIG["n_embed"] % CONFIG["n_heads"] == 0

if __name__ == "__main__":
    print("Loading libraries...")
    from training import train
    train(CONFIG)