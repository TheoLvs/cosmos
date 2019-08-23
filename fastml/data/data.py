# Usual libraries
import pandas as pd

# ML Libraries
from sklearn.model_selection import train_test_split

# Custom libraries
from .exploration import EDA



class DatasetError(Exception):
    pass


class Dataset:
    """Main class to contain ML data
    This class can be fed to EDAs, ML Problems, etc... 
    """

    def __init__(self,data,target,problem = "classification"):

        self.data = data
        self.target = target
        self.eda = EDA(self)

        if problem == "classification":
            self.data[self.target] = pd.Categorical(self.data[self.target])
        else:
            raise DatasetError("Problem value is not recognized")

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


    def train_test_split(self,test_size = 0.2,stratify = True,inplace = True):

        # Prepare stratification (split to respect target distribution in each train test group)
        stratify = self.y if stratify else None
        
        # Train test split using sklearn API
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size = test_size,stratify = stratify)

        # If inplace store as arguments
        self.splits = X_train,X_test,y_train,y_test

        if not inplace:
            return X_train,X_test,y_train,y_test



