
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class ProblemsError(Exception):
    pass


class Problem:
    def __init__(self,dataset):
        self.dataset = dataset
        self.models = {}

    def add_models(self,models):

        if models is None:
            pass

        elif isinstance(models,dict):
            self.models.update(models)
        else:
            if not isinstance(models,list): models = [models]
            models = {f"model{i+len(self.models)}_{model.__class__.__name__}":model for i,model in enumerate(models)}
            self.models.update(models)


    def _init_models(self,models = None,lr = True):

        # Prepare linear regression
        if lr:
            linear_model = self._get_linear_model()
            self.add_models({"lr":linear_model})

        # Prepare other models
        if models is not None:
            self.add_models(models)
        else:
            model = {"default_rf":self._get_default_random_forest()}
            self.add_models(model)


