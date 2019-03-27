"""
A file containing a config class and the associated hyperparametersssss
"""

config = {
    "traindatadir": "../data/train",
    "batch_size": 1,
    "val_batch_size":1,
    "learning_rate": 0.00035,
    "input_shape": (None, 14),
    "number_feature": None,
    "num_epoch": 1000,
    "checkpoint_path": "../data/output/",
    "max_to_keep": 4,
    "train_log": "../data/log/train",
    "validation_log": "../data/log/val",
    "test_log": "../data/log/test",
    "FILE_PATHS": {"dd": ("/var/www/labor_productivity_prediction/data/productivity_data.csv", "","")},
    "hidden_layer": {
        "first_layer": {
            "number":30
        }
    }
}
