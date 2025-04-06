"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os

import torch

from torchvision import transforms

# Importing the modular scripts we created earlier
import data_setup, engine, model_builder, utils

if __name__ == "__main__":
  # Setup hyperparameters
  NUM_EPOCHS = 5
  BATCH_SIZE = 32
  HIDDEN_UNITS = 10
  LEARNING_RATE = 0.001

  # Setup directories
  train_dir = "data/pizza_steak_sushi/train"
  test_dir = "data/pizza_steak_sushi/test"

  # Setup device-agnostic code 
  if torch.cuda.is_available():
      device = "cuda" # NVIDIA GPU
  elif torch.backends.mps.is_available():
      device = "mps" # Apple GPU
  else:
      device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

  print(f"Using device: {device}")

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  # This calls the `create_dataloaders()` function from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  # Create model with help from model_builder.py
  # This calls the `TinyVGG` class from model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=HIDDEN_UNITS,
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Start training with help from engine.py
  # This calls the `train()` function from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)

  # Save the model with help from utils.py
  # This calls the `save_model()` function from utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")
