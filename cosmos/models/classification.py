
# Usual libraries
import pandas as pd

# ML libraries
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix

# Custom library
from .problems import Problem,ProblemsError

class ClassificationProblem(Problem):
    """Basic classification problem in Machine Learning
    Works only in a single class setting 
    """
    
    def __init__(self,dataset,lr = True):
        """Initialization
        """
        super().__init__(dataset = dataset)
        self._init_models(lr = lr)


    @property
    def lr(self):
        """Get linear regression pre-fitted model
        """
        if hasattr(self,"models") and "lr" in self.models:
            return self.models["lr"]
        else:
            raise ProblemsError("You need to fit the linear regression first with .fit(lr = True)")
    

    def _get_linear_model(self):
        """Get linear default linear model for the classification
        """
        return LogisticRegression()


    def _get_default_random_forest(self):
        """Get default Random Forest for the classification problem
        """
        return RandomForestClassifier()


    def fit(self,models = None,X = None,y = None):
        """Fit the models using sklearn API
        """

        # Add new models to fit if any
        self.add_models(models)

        # Get default X and y from dataset object
        if X is None: X = self.dataset.X
        if y is None: y = self.dataset.y

        # Loop over each model to fit on the dataset
        for model_name in self.models:
            self.models[model_name].fit(X,y)



    def show_coefs(self,style = True):
        """Show coefficients of the fitted linear regression
        """

        # Prepare dataframe with coefficients
        coefs = pd.DataFrame({
            "feature":self.dataset.features,
            "coef":self.lr.coef_[0],
        },columns = ["feature","coef"]).sort_values("coef",ascending = False).reset_index(drop = True)
        
        # Return dataframe with or without style
        if style:
            return coefs.style.background_gradient(subset = ["coef"])
        else:
            return coefs


    def predict(self,X = None):
        """Simple predict function using the pre-fitted models
        Apply .fit() function before using predict
        
        Args:
            X (pd.DataFrame, optional): Input dataset on which to make a prediction. Defaults to None which will take the dataset X.
        
        Returns:
            pd.DataFrame: a pandas dataframe for each column the prediction by model
        """

        if X is None: X = self.dataset.X

        pred = pd.DataFrame()
        for model_name in self.models:
            model = self.models[model_name]
            pred[f"{model_name} | proba"] = model.predict_proba(X)[:,1]
            pred[f"{model_name} | class"] = model.predict(X)

        return pred


    def compute_metrics(self,pred):
        """Compute all metrics for each pre-fitted model
        """

        metrics = {}
        confusion_matrices = {}

        # Loop over each model
        for model_name in self.models:

            # Prepare columns to pull the predictions from 
            col_proba = f"{model_name} | proba"
            col_class = f"{model_name} | class"
            if col_proba not in pred.columns:
                raise ProblemsError(f"{col_proba} should be in pred")

            # Get columns to test
            y_true = pred["true"]
            y_proba = pred[col_proba]
            y_class = pred[col_class]

            # Compute metrics from sklearn API
            model_metrics = {
                "roc_auc":roc_auc_score(y_true,y_proba),
                "precision":precision_score(y_true,y_class),
                "recall":recall_score(y_true,y_class),
                "f1":f1_score(y_true,y_class),
                "accuracy":accuracy_score(y_true,y_class),
            }

            # Compute confusion matrices
            model_matrix = confusion_matrix(y_true,y_class)

            # Add computed metrics to placeholder
            metrics[model_name] = model_metrics
            confusion_matrices[model_name] = model_matrix

        # Convert dict of dict to Dataframe for easier use
        metrics = pd.DataFrame(metrics).T.style.background_gradient()

        return metrics,confusion_matrices




    def evaluate(self,fit = True,split = True,cross_validation = False,**kwargs):
        """Evaluation function for a list of models
        It fits the list of models, predict the values and evaluates the performance
        Can work with either a train test split or a cross validation
        Kwargs must be used to parameterize the split or cross val
        """

        # Evaluate models using train test split
        if split:

            # If already prepared a split, we will be using it
            if hasattr(self.dataset,"splits"):
                X_train,X_test,y_train,y_test = self.dataset.splits

            # Otherwise prepare the split using function arguments
            else:
                X_train,X_test,y_train,y_test = self.dataset.train_test_split(inplace = False,**kwargs)
            
            # Fit the models
            if fit:
                self.fit(X = X_train,y = y_train)

            # Predict the probability
            pred = self.predict(X = X_test)
            pred["true"] = y_test.values
            pred.insert(0,"index",y_test.index)

            # Compute metrics
            metrics,confusion_matrices = self.compute_metrics(pred)

            return pred,metrics,confusion_matrices

        else:
            raise ProblemsError("Cross validation is not implemented right now")
            
