import os
import sys
import tensorflow as tf
import argparse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
sys.path.append(os.environ.get("PROJECT_PATH"))

from configs import configs
from utils.preprocessing import Preprocess
from dataset import datasets
from models.model import AnnModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of dataset", type=str)
    parser.add_argument("command", help="train or test", type=str)
    parser.add_argument("--mode", help="mode name", type=str, default="")
    args = parser.parse_args()

    file_path = configs.config["FILE_PATHS"]
    train_path, validation_path = file_path[args.dataset]

    process = Preprocess(dataset=args.dataset, train_path=train_path, validation_path=validation_path,
                         config=configs.config)
    datapaths = process.preprocess()
    import pdb
    pdb.set_trace()
    ds = datasets.DataGenerator(datapaths, mode=args.command, config=process.config)
    model = AnnModel(ds.config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer)


if __name__ == '__main__':
    main()