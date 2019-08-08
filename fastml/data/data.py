
class FastMLError(Exception):
    pass



class Dataset:

    def __init__(self,data,target,problem = "classification"):

        self.data = data
        self.target = target
        self.problem = problem
        assert problem in ["classification"]

    
    @property
    def features(self):
        return self.X.columns.tolist()

    @property
    def X(self):
        return self.data.drop(columns = self.target)

    @property
    def y(self):
        return self.data[self.target]

