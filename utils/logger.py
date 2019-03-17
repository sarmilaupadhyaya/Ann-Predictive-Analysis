import tensorflow as tf


class Logger():
    def __init__(self, sess, config):

        self.config = config
        self.session = sess
        self.train_summary = tf.summary.FileWriter(self.config["train_log"])
        self.val_summary = tf.summary.FileWriter(self.config["validation_log"])
        self.test_summary = tf.summary.FileWriter(self.config["test_log"])
        self.merge_all = tf.summary.merge_all()

    def add_graph_get_writer(self, mode):
        """

        Returns:

        """
        if mode == "train":
            self.train_summary.add_graph(self.session.graph)
            return self.train_summary
        elif mode == "val":
            self.val_summary.add_graph(self.session.graph)
            return self.val_summary
        elif mode == "test":
            self.test_summary.add_graph(self.session.graph)
            return self.test_summary
