import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()
        self.init_cur_batch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config["checkpoint_path"] + "question-classification", self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        # self.saver = tf.train.import_meta_graph()
        latest_checkpoint = tf.train.latest_checkpoint(self.config["checkpoint_path"])
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_cur_batch(self):
        """

        Returns:

        """
        with tf.variable_scope('cur_batch'):
            self.current_batch = tf.Variable(0, name="current_batch", trainable=False)
            self.increment_batch_op = tf.assign(self.current_batch, self.current_batch + 1)

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
