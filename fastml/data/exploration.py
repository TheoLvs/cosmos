
# Base libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Visualisation
import missingno as msno
from ipywidgets import interact
import plotly.express as px


class EDA:
    """Exploratory Data Analysis class
    TODO:
        - Feature type inferences
        - Missing values
        - Outliers
        - Maximum occurences for categorical variables
        - Basic automatic visualisations
    """
    def __init__(self,dataset):
        self.dataset = dataset

    
    @property
    def data(self):
        return self.dataset.data


    def infer_feature_type(self,max_rows = 10000,threshold_occurences = 0.5):
        """Infer feature type (in particular categorical or numerical)
        TODO:
            - Add timestamp detection
            - More flexibily on categorical type detection
            - Split this function in two functions
        
        Args:
            max_rows (int, optional): The maximum number of rows to do the analysis, helpful for big datasets . Defaults to 10000.
            threshold_occurences (float, optional): % of unique occurences / occurences, is used to detect categorical variables. Defaults to 0.5.
        
        Returns:
            dict: a dictionary with list of variables ordered by category
        """

        columns = defaultdict(list)

        for col in self.dataset.features:
            
            # Select column as series
            column = self.dataset[col]

            # Select a subset if dataset is too long
            if len(column) > max_rows:
                column = column.sample(max_rows)

            # Get the variable type
            dtype = column.dtype

            # If type is numerical
            if dtype in [np.float64,np.int64]:
                columns["numerical"].append(col)
            else:
                if len(column.unique()) / len(column) < threshold_occurences:
                    columns["categorical"].append(col)
                else:
                    columns["big_categorical"].append(col)
                    print(f"... Too many occurences, '{col}' is probably an ID")

        return dict(columns)



    def notebook_scatter(self,**kwargs):
        """Visualize a scatter plot of interactions between variables in the notebook
        Using widgets to easily explore the data
        This function calls the associated function .show_scatter()
        """
        
        cols_x = self.data.columns
        cols_y = [self.data.columns[1],self.data.columns[0],*self.data.columns[2:]]

        @interact(
            x = cols_x,
            y = cols_y,
            hue = [self.dataset.target,*self.dataset.features,None],
            size =  [None,*self.data.columns]
        )
        def show(x,y,hue,size):
            self.show_scatter(x,y,hue,size,**kwargs)

        


    def show_scatter(self,x,y,hue = None,size = None,**kwargs):
        """Visualize a scatter plot of interactions between variables in the notebook
        See documentation here for extra arguments https://plot.ly/python/plotly-express/#scatter-and-line-plots
        In particular, marginal_x, marginal_y, trendline, facet_col, etc...
        """
        fig = px.scatter(self.data, x=x, y=y,color = hue,size = size,
            color_continuous_scale=px.colors.diverging.Tealrose,
            **kwargs)
        fig.show()



    # def show_missing_values(self,sample_size = 250):
    #     msno.matrix(self.data.sample(sample_size))
    #     plt.show()


    # def show_distribution(self,var = None,column = None,figsize = (15,4),kind = "hist"):
    #     if column is None:
    #         column = self[var]
    #     column.plot(kind = kind,figsize = figsize)
    #     if var is not None:
    #         plt.title(var)
    #     plt.show()
        
    # def show_top_occurences(self,var = None,column = None,n = 30,figsize = (15,4),kind = "bar"):
    #     if column is None:
    #         column = self[var]
            
    #     column.value_counts().head(n).plot(kind = kind,figsize = figsize)
    #     if var is not None:
    #         plt.title(var)

    #     plt.show()

