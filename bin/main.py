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
from trainers.trainer import Trainer
from utils.logger import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of dataset", type=str)
    parser.add_argument("command", help="train or test", type=str)
    parser.add_argument("--mode", help="mode name", type=str, default="")
    args = parser.parse_args()

    file_path = configs.config["FILE_PATHS"]
    train_path, validation_path, test_path = file_path[args.dataset]

    process = Preprocess(dataset=args.dataset, train=train_path, validation=validation_path,test = test_path,
                         config=configs.config)
    datapaths = process.preprocess()
    ds = datasets.DataGenerator(datapaths, mode=args.command, config=process.config)
    model = AnnModel(ds.config)
    if args.command == "train":
        with tf.Session() as sess:
            logger = Logger(config=configs.config, sess=sess)
            trainer = Trainer(model=model, data_gen=ds, session=sess, config=process.config, logger = logger)
            sess.run(tf.global_variables_initializer())
            s = sess.run(trainer.training_process())
    elif args.command == "test":
        with tf.Session() as sess:
            model.load(sess)
            logger = Logger(config=configs.config, sess=sess)
            trainer = Trainer(model=model, data_gen=ds, session=sess, config=process.config,logger = logger)
            sess.run(trainer.plot_prediction_test())


if __name__ == '__main__':
    main()