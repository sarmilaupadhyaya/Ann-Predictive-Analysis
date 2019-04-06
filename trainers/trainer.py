import tensorflow as tf
import numpy as np
from configs import configs
import pandas as pd


class Trainer:
    def __init__(self, **kwargs):
        """
        Performs the training wheel
        Args:
            model: object instance of AlexNetModel
            data_gen: object instance of DataGenerator
            config: configuration dict
            sess: session
        """
        self.__dict__.update(kwargs)

        # initializing a local version of global step, current epoch and batch
        self.global_step_value = 0
        self.current_epoch_value = 0
        self.current_batch_value = 0
        self.learning_rate = configs.config["learning_rate"]

        self.losses = []
        self.prediction = []
        self.train_writer = self.logger.add_graph_get_writer(mode="train")
        self.val_writer = self.logger.add_graph_get_writer(mode="val")

    def initialize_epoch_val(self, iterator):
        """

        Returns:

        """
        self.session.run(iterator.initializer)
        next_element = iterator.get_next()

        return next_element

    def initialize_epoch(self, iterator):
        """

        Returns:

        """
        self.session.run(iterator.initializer)
        next_element = iterator.get_next()

        self.session.run(self.model.current_batch.initializer)
        self.losses = []
        self.prediction = []
        return next_element

    def train_step(self, data_point):
        """

        Returns:

        """


        summary,prediction, self.global_step_value, self.current_batch_value, optimizer, loss = self.session.run(
                [self.logger.merge_all,self.model.prediction, self.model.global_step_tensor, self.model.increment_batch_op,
                 self.model.optimizer,
                 self.model.loss],
                feed_dict={self.model.input: data_point[0], self.model.target: data_point[1]
                           })
        self.prediction.append(prediction)

        self.losses.append(loss)

        return summary

    def val_epoch(self, val_element):
        """

        Args:
            val_element:

        Returns:

        """

        val_loss = []
        try:
            while 1:
                data_point = self.session.run(val_element)

                loss,summary,prediction,  = self.session.run(
                        [self.model.loss,self.logger.merge_all,self.model.prediction],
                        feed_dict={self.model.input: data_point[0], self.model.target: data_point[1]})

                self.val_writer.add_summary(summary,self.global_step_value)
                self.prediction.extend(prediction)
                val_loss.append(loss)

        except tf.errors.OutOfRangeError:
            return val_loss

    def train_epoch(self, next_element, val_element):
        """

        Returns:

        """

        try:

            while 1:
                data_point = self.session.run(next_element)
                summary = self.train_step(data_point)
                self.train_writer.add_summary(summary, self.global_step_value)
        except tf.errors.OutOfRangeError:
            self.train_writer.flush()
            print("Epoch: ", self.current_epoch_value)
            print("training losses : ", np.mean(self.losses))
            # self.weight_check()
            # self.weight_check()
            # print("training top_1_accuracy", np.mean(self.top_1_accuracy))
            # print("training top_2_accuracy", np.mean(self.top_2_accuracy))
            # self.weight_clipping()
            val_loss= self.val_epoch(val_element)
            self.val_writer.flush()
            # saving model
            print("validation losses : ", np.mean(val_loss))
            # print("validation top_1_accuracy", np.mean(val_top_1_accuracy))
            # print("validation top_2_accuracy", np.mean(val_top_2_accuracy))
            self.current_epoch_value = self.session.run(self.model.increment_cur_epoch_tensor)

    def training_process(self):
        """

        Returns:

        """
        iterator = self.data_gen.load(type="train")
        iterator_val = self.data_gen.load(type="val")

        while self.current_epoch_value < self.config["num_epoch"]:
            next_element = self.initialize_epoch(iterator)
            val_element = self.initialize_epoch_val(iterator_val)
            self.train_epoch(next_element, val_element)
        print('End of training')
        self.model.save(self.session)
        val_element = self.initialize_epoch_val(iterator_val)
        self.plot_prediction_test()
        return 0

    def plot_prediction(self):
        """

        :return:
        """
        iterator_val = self.data_gen.load(type="val")
        val_element = self.initialize_epoch_val(iterator_val)
        multiple = []
        losss = []
        try:
            while 1:
                data_point = self.session.run(val_element)
                single = []
                prediction,loss = self.session.run(
                        [self.model.prediction, self.model.loss],
                        feed_dict={self.model.input: data_point[0], self.model.target: data_point[1]})
                single.append(prediction)
                single.extend(data_point[1].reshape(1))
                multiple.append(single)
                losss.append(loss)

        except tf.errors.OutOfRangeError:
            pass

        print("loss: " , np.mean(losss))

        import pandas as pd
        import matplotlib.pyplot as plt
        dataframe_prediction = pd.DataFrame(multiple, columns=["Predicted","Actual"])
        dataframe_prediction["Predicted"] = dataframe_prediction.Predicted.apply(lambda x: x[0][0])
        plt.plot(dataframe_prediction["Predicted"], label ="Predicted")
        plt.plot(dataframe_prediction["Actual"],label = "Actual")
        plt.legend()
        plt.show()

    def plot_prediction_test(self):
        """

        :return:
        """
        iterator_val = self.data_gen.load(type="test")
        val_element = self.initialize_epoch_val(iterator_val)
        multiple = []
        losses = []
        data = []
        try:
            while 1:
                data_point = self.session.run(val_element)
                a = []
                single = []
                prediction, loss = self.session.run(
                        [self.model.prediction,self.model.loss],
                        feed_dict={self.model.input: data_point[0], self.model.target: data_point[1]})
                single.append(prediction)
                single.extend(data_point[1].reshape(1))
                multiple.append(single)
                losses.append(loss)
                single = [data_point[1], prediction]
                data.append(single)

        except tf.errors.OutOfRangeError:
            pass
        print("loss:",np.mean(losses) )
        import matplotlib.pyplot as plt
        dataframe_prediction = pd.DataFrame(data, columns=["Actual", "Predicted"])

        # saving dataframe
        new_data = dataframe_prediction
        dataframe_prediction["Difference"] = dataframe_prediction['Actual']- dataframe_prediction['Predicted']
        dataframe_prediction.to_csv("../data/actual_vs_predicted.csv")
        dataframe_prediction["Predicted"] = dataframe_prediction.Predicted.apply(lambda x: x[0][0])
        plt.plot(dataframe_prediction["Predicted"], label="Predicted")
        plt.plot(dataframe_prediction["Actual"], label="Actual")
        plt.legend()
        plt.show()

    def normailize(self, value, mean, std):

        train = (value - mean) / std
        return train

    def analysis(self):


        df = pd.read_csv("../data/productivity_data.csv")
        df["Labor percent"] = df["Labor percent"].apply(lambda x: float(x) / 100)
        load_data = df.dropna()
        labels = np.expand_dims(load_data["Actual Productivity (m3/hr)"].as_matrix().astype(np.float32), axis=1)
        # normalization
        for column in list(load_data.columns):
            if column != "Actual Productivity (m3/hr)":
                numpy_format = load_data[column].as_matrix()
                mean, std = numpy_format.mean(), numpy_format.std()
                load_data[column] = load_data[column].apply(lambda x: self.normailize(x, mean, std))
        for column in list(load_data.columns):
            a = []
            b = []
            if column != "Actual Productivity (m3/hr)":
                min = load_data[column].min()
                max = load_data[column].max()
                row = load_data.iloc[20]
                i = 0.1
                for __ in range(20):
                    row[column] = i
                    a.append(i)
                    X = np.array(list(row)[:15]).reshape([1,15])
                    prediction = self.session.run( self.model.prediction,
                        feed_dict={self.model.input: X})
                    b.append(prediction.tolist()[0][0])
                    i+= 0.05
            temp = pd.DataFrame([], columns = ["Actual Productivity (m3/hr)",column])
            temp["Actual Productivity (m3/hr)"] = pd.Series(b)

            import pdb
            pdb.set_trace()
            if column in ["Floor height (ft)","Alterations in design or drawings", "Improvement in construction method"]:
                a = a[::-1]

            temp[column] = pd.Series(a)


            import matplotlib.pyplot as plt
            temp.plot(x = column, y = "Actual Productivity (m3/hr)")
            plt.show()









