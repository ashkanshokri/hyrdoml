# Import hydroml components
from hydroml.data.dataset import Dataset
from hydroml.data.catchment import Catchment
from hydroml.models.get_model_from_path import get_model_from_path
from hydroml.prediction.prediction import process_and_convert_dataloader_to_xarray
import xarray as xr

from hydroml.models.get_model_from_path import get_model_from_path



def get_required_data_dims(model_path):
    model = get_model_from_path(model_path)
    config = model.config
    return {'dynamic_data': config.lstm_dynamic_input_feature_size, 'static_data': config.lstm_static_input_size}


def run_hydrological_simulation(model_path, dynamic_data, static_data, catchment_id, clip_at_zero=True, **kwargs) -> xr.Dataset:
    """
    Runs a hydrological simulation using the given model path, dynamic data, and static data.

    Parameters:
    - model_path (str): Path to the model.
    - dynamic_data (pd.DataFrame): Time-indexed dynamic forcing data (e.g., precipitation, temperature).
    - static_data (list): Static features for the catchment (e.g., catchment area, mean slope).
    - catchment_id (str): Identifier for the catchment.
    - start_date (str): Start date for the simulation (default: '2024-01-01').

    Returns:
    - pd.Series: Simulation results as a time series of predictions.
    """
    
    
    # Load the model
    model = get_model_from_path(model_path)
        
    model.config.parent_path = str(model_path.parent.parent)
    model.config.name = model_path.parent.name
    model.config.version = model_path.name
    #model.config.transform_parameter_path = 'params.yaml'

    config = model.config

    # update the transform parameter path. if the model is calibrated on another system, for simulation it needs path to the parameters get updated
    #config.transform_parameter_path = model_path.parent / 'params.yaml'

    if isinstance(static_data, dict):
        static_data = [static_data[x] for x in config.static_features]

    # Set up configuration for testing
    config.test = {
        'catchment_ids': [catchment_id],
        'periods': [[dynamic_data.index[0].strftime('%Y-%m-%d'), dynamic_data.index[-1].strftime('%Y-%m-%d')]]
    }

    # Create Catchment object
    catchments = [Catchment(dynamic_data, static_data, id=catchment_id, metadata={})]

    # if the model is calibrated on another system, for simulation it needs to have the paths to the required data
    config.update(kwargs)

    # Create Dataset object
    dataset = Dataset.from_catchments(catchments, config, 'test', is_train=False)

    # Convert Dataset to DataLoader
    dataloader = dataset.to_dataloader()

    # Process DataLoader and get predictions
    ds = process_and_convert_dataloader_to_xarray(dataloader, model, config, clip_at_zero=clip_at_zero)

    # Convert predictions to pandas DataFrame
    #df = ds.to_dataframe().reset_index().set_index('date')['prediction']

    return ds


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Example inputs
    model_path = 'P://work//sho108//hydroml//results//default//241209224715_cdae'

    length = 300
    dynamic_data = pd.DataFrame(
        data=np.random.rand(length, 2) * 100,  # Example dynamic data
        columns=['precipitation_AWAP', 'et_morton_wet_SILO'],
        index=pd.date_range(start='2024-01-01', periods=length, freq='D')
    )
    static_data = [1000, 10, 50, 5, 1, 2, 0.1, 0.5, 800, 20, 0.2, 15, 5]  # Example static features
    catchment_id = '000000'

    # Run the simulation
    simulation = run_hydrological_simulation(model_path, dynamic_data, static_data, catchment_id)

    # Display the simulation results
    simulation.plot(figsize=(10, 3))
    plt.title("Hydrological Simulation")
    plt.xlabel("Date")
    plt.ylabel("Prediction")
    plt.show()
