# imports
from hydroml.config.config import Config
import torch.nn as nn
import torch
from typing import Dict, Any

# functions
def get_loss_fn(config: Config) -> nn.Module:
    """Get the loss function based on the configuration.

    Args:
        config (Config): Configuration object containing loss function type.

    Returns:
        nn.Module: The loss function to be used.
    
    Raises:
        ValueError: If the specified loss function is not supported.
    """
    if config.loss_fn == 'mse':
        return mse_loss
    elif config.loss_fn == 'nse':
        return nse_loss
    else:
        raise ValueError(f"Loss function {config.loss_fn} not supported")


def mse_loss(prediction: torch.Tensor, ground_truth: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    """Calculate the Mean Squared Error (MSE) loss.

    Args:
        prediction (torch.Tensor): The predicted values.
        ground_truth (torch.Tensor): The actual values.
        metadata (Dict[str, Any]): Additional metadata containing weights and observed variance.

    Returns:
        torch.Tensor: The computed MSE loss.
    """
    observed_variance = metadata['observed_variance'].unsqueeze(1).unsqueeze(1)
    
    weight = metadata.get('weight', torch.ones_like(observed_variance)).unsqueeze(1).unsqueeze(1)

    mask = ~torch.isnan(ground_truth)

    masked_prediction = prediction[mask]
    masked_ground_truth = ground_truth[mask]
    masked_weight = weight[mask]

    loss = masked_weight * (masked_prediction - masked_ground_truth) ** 2
    return torch.mean(loss)


def nse_loss(prediction: torch.Tensor, ground_truth: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    """Calculate the Nash-Sutcliffe Efficiency (NSE) loss.

    Args:
        prediction (torch.Tensor): The predicted values. Expected shape: (batch_size, seq_len, target_size)
        ground_truth (torch.Tensor): The actual values. Expected shape: (batch_size, seq_len, target_size)
        metadata (Dict[str, Any]): Additional metadata containing weights and observed variance. Expected shape: (batch_size)

    Returns:
        torch.Tensor: The computed NSE loss.
    """
    # convert all the tensors to the same shape - (batch_size, time - usually  1, target_size usually 1)
    observed_variance = metadata['observed_target_std']
    weight = metadata.get('weight', torch.ones_like(observed_variance))

    # match the size of observed variance and weight to the prediction
    
    observed_variance = observed_variance.unsqueeze(1).repeat(1, prediction.size(1), 1)
    
    if len(weight.shape) == 1:
        weight = weight.unsqueeze(1).unsqueeze(1)
    elif len(weight.shape) == 2:
        weight = weight.unsqueeze(1)
    else:
        raise ValueError(f"Weight shape {weight.shape} is not supported")

    weight = weight.repeat(1, prediction.size(1), 1)
    
    # mask the nan values - all data will be converted to a 1d tensor
    mask = ~torch.isnan(ground_truth) 
    masked_prediction = prediction[mask] 
    masked_ground_truth = ground_truth[mask]   
    masked_observed_variance = observed_variance[mask]
    masked_weight = weight[mask]
    

    # resedual calculations. All variable are in a same shape as masked variables
    squared_error = (masked_prediction - masked_ground_truth) ** 2 
    scaled_loss = masked_weight / (masked_observed_variance + 1) * squared_error
    adjusted_loss = torch.where(scaled_loss > 1000, 1000 + (1000 - scaled_loss) * 0.1, scaled_loss)
    

    # return the mean of the adjusted loss
    return torch.mean(adjusted_loss)

