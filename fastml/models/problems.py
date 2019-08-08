
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class FastMLError(Exception):
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



class ClassificationProblem(Problem):
    def __init__(self,dataset):
        super().__init__(dataset = dataset)
        print("... Fitting default models")
        self.fit()

    @property
    def lr(self):
        if hasattr(self,"models") and "lr" in self.models:
            return self.models["lr"]
        else:
            raise FastMLError("You need to fit the linear regression first with .fit(lr = True)")
    
    def _get_linear_model(self):
        return LogisticRegression()

    def _get_default_random_forest(self):
        return RandomForestClassifier()

    def fit(self,models = None,lr = True,X = None,y = None):
        self.models = self._init_models(models = models,lr = lr)
        if X is None: X = self.dataset.X
        if y is None: y = self.dataset.y

        for model_name in self.models:
            self.models[model_name].fit(X,y)


    def show_coefs(self,style = True):
        coefs = pd.DataFrame({
            "feature":self.dataset.features,
            "coef":self.lr.coef_[0],
        },columns = ["feature","coef"]).sort_values("coef",ascending = False).reset_index(drop = True)
        if style:
            return coefs.style.background_gradient(subset = ["coef"])
        else:
            return coefs


    def predict(self,proba = False,X = None):

        if X is None: X = self.dataset.X

        pred = pd.DataFrame()
        for model_name in self.models:
            model = self.models[model_name]
            if proba:
                pred[model_name] = model.predict_proba(X)[:,1]
            else:
                pred[model_name] = model.predict(X)

        return pred


