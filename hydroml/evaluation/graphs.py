import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from typing import Union, Dict

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

def exceedance_curve(data: pd.Series, ax:mpl.axes._axes.Axes=None, **kwargs):
    """
    Plot the exceedance probability of the data on the given axes.

    Parameters:
    data (pd.Series): The data series to plot.
    ax (matplotlib.axes._axes.Axes): The matplotlib axes to plot on.

    Returns:
    None
    """
    
    ax = ax or  plt.gca()
    # Sort the data in descending order
    sorted_data = np.sort(data)[::-1]

    # Calculate the exceedance probability
    exceedance_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Plot the data
    p = ax.plot(exceedance_prob, sorted_data, **kwargs)

    # Set the labels and title
    ax.set_xlabel('Exceedance Probability')
    ax.set_ylabel('Value')
    ax.set_title('Exceedance Probability Plot')

    # Optionally set logarithmic scale if the data spans several orders of magnitude
    # ax.set_yscale('log')
    # ax.set_xscale('log')

    # Show grid
    ax.grid(True)

    return p






def spatial_plot(shpfile: Union[str, Path],
                 metrics_series: pd.Series,
                 shp_basin_field: str,
                 **kwargs) -> None:
    """
    Visualize the results of a neural hydrology model evaluation on a map.

    Parameters:
    - shpfile (Union[str, Path]): Path to the shapefile containing basin boundaries.
    - results (Dict[str, Dict[str, pd.DataFrame]]): Output of the neural hydrology model evaluation.
    - shp_basin_field (str): Field in the shapefile representing basin IDs (default is 'CatchID').
    - metric (str): The evaluation metric to visualize (default is 'NSE').
    - **kwargs: Additional keyword arguments for the plot function.

    Returns:
    None
    """
    # Set default values for kwargs
    default_kwargs = dict(legend=True, edgecolor='k', linewidth=0.3)
    kwargs = {**default_kwargs, **kwargs}

    # Read the shapefile
    gdf = gpd.read_file(shpfile)



    metrics_series.name = metrics_series.name or 'value'
    # Merge the shapefile with the metric dataframe

    gdf_merged = pd.merge(gdf, metrics_series, left_on=shp_basin_field, right_index=True)
    gdf_merged.plot(column=metrics_series.name, **kwargs)


def spatial_plot_categorical(shpfile: Union[str, Path],
                 metrics_series: pd.Series,
                 shp_basin_field: str,
                 category_order: list = None,
                 **kwargs) -> None:
    """
    Visualize the results of a neural hydrology model evaluation on a map.

    Parameters:
    - shpfile (Union[str, Path]): Path to the shapefile containing basin boundaries.
    - results (Dict[str, Dict[str, pd.DataFrame]]): Output of the neural hydrology model evaluation.
    - shp_basin_field (str): Field in the shapefile representing basin IDs (default is 'CatchID').
    - metric (str): The evaluation metric to visualize (default is 'NSE').
    - **kwargs: Additional keyword arguments for the plot function.

    Returns:
    None
    """
    import geopandas as gpd
    # Set default values for kwargs
    default_kwargs = dict(legend=True, edgecolor='k', linewidth=0.3)
    kwargs = {**default_kwargs, **kwargs}

    # Read the shapefile
    gdf = gpd.read_file(shpfile)



    metrics_series.name = metrics_series.name or 'value'
    # Merge the shapefile with the metric dataframe

    gdf_merged = pd.merge(gdf, metrics_series, left_on=shp_basin_field, right_index=True)
    if category_order is not None:
        gdf_merged[metrics_series.name] = pd.Categorical(gdf_merged[metrics_series.name], categories=category_order, ordered=True)
    # Plot the results on the map

    
    return gdf_merged.plot(column=metrics_series.name, **kwargs)



def add_boundary(shp, **kwargs):
    
    gdf = gpd.read_file(shp)
    
    gdf.plot( **kwargs) 



def scatter(metrics, catchment_area, xmin=-0.5, ymin=-0.5, xmax=1, ymax=1):
    import seaborn as sns

    # Defining the categories
    def categorize_catchment_area(area):
        if area <= 1000:
            return 'small'
        else:
            return 'Larger'

    data = pd.concat([metrics, catchment_area], axis=1).dropna()
    # Applying the categorization
    data['catchment_category'] = data['catchment_area'].apply(categorize_catchment_area)

    data['awra'] = data['awra'].clip(xmin, xmax)
    data['lstm_ub'] = data['lstm_ub'].clip(ymin, ymax)

    scatter_plot = sns.scatterplot(data, x='awra', y ='lstm_ub', hue='catchment_category', size=1, alpha=0.75)

    # Adding the means
    handles, labels = scatter_plot.get_legend_handles_labels()
    category_colors = {label: handle.get_color() for handle, label in zip(handles, labels)}


    for category in data['catchment_category'].unique():
        category_data = data[data['catchment_category'] == category]
        mean_awra = category_data['awra'].mean()
        mean_lstm_ub = category_data['lstm_ub'].mean()
        plt.scatter(mean_awra, mean_lstm_ub, marker='x', s=200, color=category_colors[category], label=f'{category} mean')


    plt.plot([-10,1], [-10,1], '--r')
    plt.xlim(xmin-0.01,xmax)
    plt.ylim(ymin-0.01,ymax)
    plt.gca().set_aspect('equal')