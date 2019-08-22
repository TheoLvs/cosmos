import pandas as pd
from .exploration import EDA


class FastMLError(Exception):
    pass



class Dataset:

    def __init__(self,data,target,problem = "classification"):

        self.data = data
        self.target = target
        self.eda = EDA(self)

        if problem == "classification":
            self.data[self.target] = pd.Categorical(self.data[self.target])

    def __getitem__(self,key):
        return self.data[key]


    def _repr_html_(self):
        return self.data.head(20)._repr_html_()

    
    @property
    def features(self):
        return self.X.columns.tolist()

    @property
    def X(self):
        return self.data.drop(columns = self.target)

    @property
    def y(self):
        return self.data[self.target]

