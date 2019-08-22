
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class ProblemsError(Exception):
    pass


class Problem:
    def __init__(self,dataset):
        self.dataset = dataset

    def _init_models(self,models = None,lr = True):

        models_to_fit = {}

        # Prepare linear regression
        if lr:
            linear_model = self._get_linear_model()
            models_to_fit.update({"lr":linear_model})

        # Prepare other models
        if models is not None:
            if not isinstance(models,list): models = [models]
            models = {f"model{i}":model for i,model in enumerate(models)}
            models_to_fit.update(models)
        else:
            model = {"default_rf":self._get_default_random_forest()}
            models_to_fit.update(model)

        return models_to_fit


