import os
import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def load_csv(self):
        pass

    def normailize(self, value, mean, std):

        train = (value - mean) / std
        return train



    def preprocess(self):
        """

        :return:
        """
        numpy_path = {}
        for x in ["train", "validation", "test"]:
            path = eval("self." + x)
            each_path = "../data/" + x + "/" + x + "_numpy.npz"
            result = os.path.exists(each_path)
            if result:
                pass
            elif result == False and path != "":
                load_data = pd.read_csv(path)
                import pdb
                pdb.set_trace()
                load_data["Labor percent"] = load_data["Labor percent"].apply(lambda x: float(x) / 100)
                load_data = load_data.dropna()
                labels = np.expand_dims(load_data["Actual Productivity (m3/hr)"].as_matrix().astype(np.float32), axis=1)
                del load_data["Actual Productivity (m3/hr)"]
                print(len(list(load_data.columns)))
                # normalization
                print(len(load_data))
                for column in list(load_data.columns):
                    numpy_format = load_data[column].as_matrix()
                    mean, std = numpy_format.mean(), numpy_format.std()
                    load_data[column] = load_data[column].apply(lambda x : self.normailize(x, mean, std))
                numpy_data = load_data.as_matrix().astype(np.float32)
                np.savez(each_path, numpy_data, labels)
            else:
                each_path = ""
            numpy_path[x] = each_path

        return numpy_path

# here csv file will be loaded and preprocess and finally converted to numpy file and saved in npz file
