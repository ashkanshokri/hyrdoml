# imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydroml.config.config import Config
from hydroml.training.loss import get_loss_fn
from torch.optim.lr_scheduler import LambdaLR
from hydroml.models.fully_connected import FullyConnected as FC
from typing import Dict, Union, Any


# class
class HydroLSTM(pl.LightningModule):
    """LSTM model for time series forecasting using PyTorch Lightning.

    Parameters
    ----------
    config : Config
        Configuration object containing model parameters and hyperparameters.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config: Config = config

        # Define model parameters
        self.lstm_dynamic_input_feature_size: int = config.lstm_dynamic_input_feature_size
        self.lstm_dynamic_input_feature_latent_size: int = config.lstm_dynamic_input_feature_latent_size
        self.lstm_static_input_size: int = config.lstm_static_input_size
        self.lstm_static_input_latent_size: int = config.lstm_static_input_latent_size

        # Define model architecture
        self.lstm_hidden_size: int = config.lstm_hidden_size
        self.lstm_target_size: int = config.lstm_target_size
        self.lstm_layers: int = config.lstm_layers
        self.lr: float = config.lr

        

        # Initialize static embedding layer
        if self.lstm_static_input_size > 0:
            self.static_embedding: nn.Linear = nn.Linear(self.lstm_static_input_size, self.lstm_static_input_latent_size)
            lstm_input_size: int = self.lstm_dynamic_input_feature_latent_size + self.lstm_static_input_latent_size
        else:
            self.static_embedding: nn.Identity = nn.Identity()
            lstm_input_size: int = self.lstm_dynamic_input_feature_latent_size

        self.dynamic_embedding: nn.Linear = nn.Linear(self.lstm_dynamic_input_feature_size, self.lstm_dynamic_input_feature_latent_size)

        # Create the main LSTM model
        self.lstm: nn.LSTM = nn.LSTM(input_size=lstm_input_size,
                                      hidden_size=self.lstm_hidden_size,
                                      num_layers=self.lstm_layers,
                                      batch_first=True)

        # Initialize dropout layer
        self.dropout: Union[nn.Dropout, nn.Identity] = nn.Dropout(config.dropout_probability) if config.dropout_probability > 0 else nn.Identity()

        # Initialize the head of the model
        n_in: int = self.lstm_hidden_size
        n_out: int = self.lstm_target_size
        layers = [FC(n_in, config.head_hidden_layers + [n_out], 'linear')]

        self.head = nn.Sequential(*layers)

        self.loss_fn = get_loss_fn(config)

        # Save hyperparameters for logging
        self.save_hyperparameters()  # this is needed so load_from_checkpoint does not need config as args.
        self._reset_parameters()


    def freeze_layers(self, layers: list):
        """Freeze parts of the model."""

        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = False
        
    def unfreeze_layers(self, layers: list):
        """Unfreeze parts of the model."""
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = True

    def freeze_all_layers(self):
        """Freeze all layers of the model."""
        self.freeze_layers(['static_embedding', 'dynamic_embedding', 'lstm', 'head'])

    def unfreeze_all_layers(self):
        """Unfreeze all layers of the model."""
        self.unfreeze_layers(['static_embedding', 'dynamic_embedding', 'lstm', 'head'])

    def _reset_parameters(self) -> None:
        """Reset the parameters of the LSTM model."""
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                # Set forget gate bias (second quarter of the vector)
                n = param.size(0) // 4
                param.data[n:2*n] = self.config.initial_forget_bias

    def get_forget_bias(self):
        """Get the forget bias of the LSTM model."""
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0) // 4
                return param.data[n:2*n]

    @staticmethod
    def merge_static_dynamic(dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Merge static and dynamic features.

        Args:
            dynamic (torch.Tensor): Dynamic features tensor.
            static (torch.Tensor): Static features tensor.

        Returns:
            torch.Tensor: Merged tensor of dynamic and static features.

            Alternative mthod:
            repeated_static = static.unsqueeze(-2).expand_as(dynamic)
            return torch.cat((dynamic, repeated_static), dim=-1)
        """
        
        if len(dynamic.shape) == 2:
            repeated_encoded_static_data = static.repeat(dynamic.shape[-2],1)
        else:
            repeated_encoded_static_data = static.unsqueeze(1).repeat(1,dynamic.shape[-2],1)
        
        return torch.cat((dynamic, repeated_encoded_static_data), dim=-1)
    


    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x_dynamic (torch.Tensor): Dynamic input features.
            x_static (torch.Tensor): Static input features.

        Returns:
            torch.Tensor: Model predictions.
        """
        # Embed dynamic features
        x_dynamic = self.dynamic_embedding(x_dynamic)
        x_dynamic = self.dropout(x_dynamic)  # Dropout after embedding

        # Concatenate dynamic and static features and make the tensor to be passed to LSTM
        if self.lstm_static_input_size > 0:
            x_static = self.static_embedding(x_static)
            x_static = self.dropout(x_static)  # Dropout after embedding static
            

            x = self.merge_static_dynamic(x_dynamic, x_static)
            x = self.dropout(x)  # Dropout after merging
        else:
            x = x_dynamic
        
        # Pass to LSTM
        x, (h, c) = self.lstm(x)
        x = self.dropout(x)  # Dropout after LSTM

        # Pass to head

        y = self.head(x[..., -1 * self.config.number_of_time_output_timestep:, :])

        return y

    def loss(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Calculate the loss for the given batch.

        Args:
            batch (Dict[str, Any]): Batch of data containing inputs and targets.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        prediction = self(batch['x_dynamic'], batch['x_static'])
        ground_truth = batch['y']
        metadata = batch['metadata']
        
        return self.loss_fn(prediction, ground_truth, metadata)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step for the model.

        Args:
            batch (Dict[str, Any]): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        loss = self.loss(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step for the model.

        Args:
            batch (Dict[str, Any]): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the validation loss.
        """
        loss = self.loss(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: Union[int, None] = None) -> tuple:
        """Prediction step for the model.

        Args:
            batch (Dict[str, Any]): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (Union[int, None], optional): Index of the dataloader. Defaults to None.

        Returns:
            tuple: Tuple containing indices, predictions, ground truth, static features, metadata, and dates. A 3d tensor with shape (batch_size, time, target_size) usually (batch_size, 1, 1)
        """
        prediction = self(batch['x_dynamic'], batch['x_static'])
        #prediction = self(batch['x_dynamic'], batch['x_static'])
        #ground_truth = batch['y']
        #metadata = batch['metadata']
        return prediction

    def configure_optimizers(self) -> tuple:
        """Configure optimizers and learning rate schedulers.

        Returns:
            tuple: Tuple containing the optimizer and the scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.config.weight_decay)
        
        # Define a function to adjust learning rate based on current step
        def lr_lambda(current_step: int) -> float:
            return max(1, 10 - current_step * 0.1)
        
        # Create a scheduler that adjusts learning rate using the above function
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [scheduler]
