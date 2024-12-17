import numpy as np
import xarray as xr
import pandas as pd


def get_metrics(ds):

    return Metrics(ds['y'], ds['prediction']).all_metrics()

class Metrics:
    def __init__(self, observed, predicted):
        """
        observed and predicted are xarray DataArrays with 'date' and 'catchment_id' dimensions.
        """
        if not isinstance(observed, xr.DataArray) or not isinstance(predicted, xr.DataArray):
            raise TypeError("Inputs must be xarray.DataArrays.")
        
        # Ensure dimensions match
        if observed.dims != predicted.dims:
            raise ValueError("Observed and predicted must have the same dimensions.")
        
        # Ensure alignment
        self.observed = observed
        self.predicted = predicted
        self._residuals = None
        self._squared_residuals = None
        self._observed_deviation = None
        self._residuals_sqrt = None
        self._squared_residuals_sqrt = None
        self._observed_deviation_sqrt = None
        
    def residuals(self):
        """
        Helper function to calculate residuals, with caching.
        """
        if self._residuals is None:
            self._residuals = self.observed - self.predicted
        return self._residuals
    
    def residuals_sqrt(self):
        """
        Helper function to calculate residuals, with caching.
        """
        if self._residuals_sqrt is None:
            self._residuals_sqrt = np.sqrt(self.observed) - np.sqrt(self.predicted)
        return self._residuals_sqrt
    
    def squared_residuals(self):
        """
        Helper function to calculate squared residuals, with caching.
        """
        if self._squared_residuals is None:
            self._squared_residuals = self.residuals() ** 2
        return self._squared_residuals
    
    def squared_residuals_sqrt(self):
        """
        Helper function to calculate squared residuals in sqrt space, with caching.
        """
        if self._squared_residuals_sqrt is None:
            self._squared_residuals_sqrt = self.residuals_sqrt() ** 2
        return self._squared_residuals_sqrt
    
    def observed_deviation_sqrt(self):
        """
        Helper function to calculate observed deviation in sqrt space, with caching.
        """
        if self._observed_deviation_sqrt is None:
            self._observed_deviation_sqrt = np.sqrt(self.observed) - np.sqrt(self.observed.mean(dim="date"))
        return self._observed_deviation_sqrt
    
    def observed_deviation(self):
        """
        Helper function to calculate observed deviation, with caching.
        """
        if self._observed_deviation is None:
            self._observed_deviation = self.observed - self.observed.mean(dim="date")
        return self._observed_deviation

    def nse(self):
        """
        Calculate Nash-Sutcliffe Efficiency (NSE) per catchment_id.
        """
        
        numerator = (self.squared_residuals()).mean(dim="date")
        denominator = (self.observed_deviation() ** 2).mean(dim="date")
        return 1 - (numerator / denominator)

    def kge(self):
        """
        Calculate Kling-Gupta Efficiency (KGE) per catchment_id.
        """
        obs_mean = self.observed.mean(dim="date")
        pred_mean = self.predicted.mean(dim="date")
        
        # Correlation (r)
        correlation = xr.corr(self.observed, self.predicted, dim="date")
        
        # Bias ratio (β)
        bias_ratio = pred_mean / obs_mean
        
        # Variability ratio (γ)
        obs_std = self.observed.std(dim="date")
        pred_std = self.predicted.std(dim="date")
        variability_ratio = pred_std / obs_std
        
        # KGE calculation
        kge = 1 - np.sqrt((correlation - 1) ** 2 + (bias_ratio - 1) ** 2 + (variability_ratio - 1) ** 2)
        return kge

    def rmse(self):
        return np.sqrt(self.squared_residuals().mean(dim="date"))
    
    def bias(self):
        return self.predicted.mean(dim="date") / self.observed.mean(dim="date")
    
    def relative_bias(self):
        return (self.predicted.mean(dim="date") - self.observed.mean(dim="date")) / self.observed.mean(dim="date")

    def absolute_bias(self):
        return np.abs(self.bias())
    
    def nse_sqrt(self):
        '''
        nse in sqrt space'''
        numerator = (self.squared_residuals_sqrt()).mean(dim="date")
        denominator = (self.observed_deviation_sqrt() ** 2).mean(dim="date")
        return 1 - (numerator / denominator)




        
    def all_metrics(self):
        return xr.Dataset({
            'nse': self.nse(), 
            'kge': self.kge(),
            'rmse': self.rmse(),
            'bias': self.bias(),
            'relative_bias': self.relative_bias(),
            'absolute_bias': self.absolute_bias(),
            'nse_sqrt': self.nse_sqrt()
        })
