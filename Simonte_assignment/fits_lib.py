import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import bisect
from statsmodels.stats.outliers_influence import variance_inflation_factor


def n_components(evr, threshold):

    """ The functions finds the number of components needed 
        to reproduce a certain percentage (dictated by the 
        threshold) of the total variance of data.
             
        Params
        evr: ndarray
        explained variance ratio of the n component

        threshold: float
        How much of the total n-dimensional variance
        one wants to reproduce.
                
        Outputs:
        n: int
        Return the number of principal components needed to reproduce 
        a certain percentage (dictated by the threshold) of the 
        total variance
    """
    # print(np.cumsum(evr),threshold)
    return bisect.bisect_left(np.cumsum(evr),threshold)


def principal_component_analysis(x, threshold):

    """ Principal component analysis (PCA). It is useful
        when dealing with correlated "independent" variables (X).
        With this method is possible to understand in which
        "direction" the X change the most. 
        If the predictors are not on the same scale, the columns
        of matrix X should be standardised before implementing PCA.
                
             
        Params
        x: correlated independent variables
        threshold: float
        How much of the total n-dimensional variance
        one wants to reproduce within the first N components.
                
        Outputs:
        x_pca: array
        return the principal components needed to reproduce a 
        certain percentage (dictated by the threshold) of the 
        total variance

        References
        https://bookdown.org/ssjackson300/Machine-Learning-Lecture-Notes/pcr.html
        https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit_transform
    """
            
    # scaling to mean=0 and variance=1 to equally compare the data
    x_scaled = StandardScaler().fit_transform(x)
            
    # find the percentage of the total variance explained by each components
    pca = PCA().fit(x_scaled)
    n = n_components(pca.explained_variance_ratio_,threshold)
            
    # find the n components
    pca = PCA(n_components=n+1)
    x_pca = pca.fit_transform(x_scaled)

    return x_pca

class data_conversion:
    
    def __init__(self, df=pd.DataFrame()):
        self.data = df
    
    def ordinal_to_numeric(self, column, rank):

        """ Conversion from ordinal to numeric data. 

            Params
            column: string
                    column of the dataframe to be converted 
            rank: list
                  chosen rank used to order the data
            output : pandas Series
                     Return the converted series of the dataframe

            References
            https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html
         """
        
        self.data[column] = pd.Categorical(self.data[column], rank, ordered=True)
        return self.data[column].cat.codes

    def categorical_to_numeric(self, column):

        """ Conversion from categorical to numeric data
            using dummy variables (0/1).
             
            Params
            column: string
                    column of the dataframe to be converted 

            output : pandas Series
                     Return the converted series of the dataframe

            References
            https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
         """

        self.data[column] = pd.get_dummies(self.data[column], prefix=column, dtype='int', drop_first=True)

        return self.data[column]

    def data_cleaning(self):

        """ Clean data from non finite values
                
            Outputs
            cleaned_df: dataframe

            cleaned dataframes
        """
            
        # remove non-finite numbers from dataframe
        cleaned_df = self.data.loc[np.isfinite(self.data).all(axis=1)]

        return cleaned_df

    def calc_vif(self):
        
        """ Variance Inflation Factor (VIF) determines the 
            strength of the correlation between the 
            independent variables. It is predicted by 
            taking a variable and regressing it 
            against every other variable. 
            One recommendation is that if VIF of a variable
            is greater than 5, then the variable is highly 
            collinear with other variables.

            Outputs:
            vif: dataframe
                 
                 dataframe containing the name of the indipendent 
                 variables and relative VIF.

            References:
            https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
        """
        # Calculating VIF
        vif = pd.DataFrame()
        vif["variables"] = self.data.columns
        vif["VIF"] = [variance_inflation_factor(self.data.values, i) for i in range(self.data.shape[1])]

        return(vif)


class linear_regression:

        def __init__(self, independent_variable, dependent_variable):
            self.x = independent_variable
            self.y = dependent_variable

        def regression(self, corr_x=True):

            """ Linear regression computed by minimising
                the residual sum of squares
                
                Params
            
                corr_x: bool
                        Specify whether there correlated independent 
                        variables and PCA must be performed.
                        Default: False
                
                Outputs:
                        self: object

                References:
                https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
            """

            # apply pca in case of correlated x variables
            if corr_x: x_dataset = principal_component_analysis(self.x, 0.95)
            else: x_dataset = self.x
            
            reg = LinearRegression().fit(x_dataset, self.y)

            return reg

