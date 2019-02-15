class Preprocess:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def load_csv(self):
        pass

    def preprocess(self):
        return {"train":"", "test":""}

# here csv file will be loaded and preprocess and finally converted to numpy file and saved in npz file
