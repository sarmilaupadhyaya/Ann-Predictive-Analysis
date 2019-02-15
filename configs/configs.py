"""
A file containing a config class and the associated hyperparametersssss
"""

config = {
    "traindatadir": "../data/train",
    "batch_size": 20,
    "learning_rate": 0.0001,
    "input_shape": (None, "num_feature"),
    "number_feature": None,
    "num_epoch": 50,
    "checkpoint_path": "../data/output/",
    "max_to_keep": 4,
    "train_log": "../data/log/train",
    "validation_log": "../data/log/val",
    "input_shape": (None, 7),
    "FILE_PATHS": {"dd": ("ggg", "")},
    "hidden_layer": {
        "first_layer": {
            "number":8
        }
    }
}
