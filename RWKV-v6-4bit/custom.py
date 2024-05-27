import torch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
import lightning
from pytorch_lightning import Callback
import os

class SaveElementWeightsCallback(lightning.pytorch.callbacks.callback.Callback):
    def __init__(self, save_dir, elements_to_save):
        super().__init__()
        self.save_dir = save_dir
        self.elements_to_save = elements_to_save
    def on_train_epoch_end(self, trainer, pl):
        # Extract the desired elements from the model's state_dict
        model_state_dict = pl.state_dict()
        #saved_weights = {k: v for k, v in model_state_dict.items() if k in self.elements_to_save}
        saved_weights = {
            k: v for k, v in model_state_dict.items()
            if any(elem in k for elem in self.elements_to_save)
        }

        # Print all keys in saved_weights
        print("Keys of saved weights:")
        for key in saved_weights.keys():
            print(key)

        # Ensure the directory exists
        os.makedirs(self.save_dir, exist_ok=True)  # Add this line

        # Save the weights to the specified directory
        epoch = trainer.current_epoch + 1
        filename = f"epoch_{epoch}_weights.pth"
        torch.save(saved_weights, os.path.join(self.save_dir, filename))
