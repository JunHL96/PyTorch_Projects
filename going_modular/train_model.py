import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Trains the model for one epoch and returns average loss and accuracy.
    Uses a tqdm progress bar to show per-batch metrics.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    batch_pbar = tqdm(dataloader, desc="Training", leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches")
    for X, y in batch_pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(y_pred, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

        batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def eval_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model for one epoch and returns average loss and accuracy.
    Uses a tqdm progress bar to display per-batch metrics.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    batch_pbar = tqdm(dataloader, desc="Validation", leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches")
    with torch.inference_mode():
        for X, y in batch_pbar:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(y_pred, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                epochs: int,
                device: torch.device,
                base_lr: float,
                patience: int = 3,
                unfreeze_epoch: int = 3,
                unfreeze_option: str = "last") -> Dict[str, List]:
    """
    Trains the model for a number of epochs with early stopping and gradual unfreezing.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        epochs: Total number of epochs.
        device: Target device (cuda, mps, cpu).
        base_lr: Base learning rate (used when adjusting learning rate upon unfreezing).
        patience: Number of epochs with no improvement before early stopping.
        unfreeze_epoch: Epoch at which to gradually unfreeze additional layers.
        unfreeze_option: Which set of layers to unfreeze ("last", "last_two", etc.).

    Returns:
        A dictionary containing the training and validation history for loss and accuracy.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0

    epoch_pbar = tqdm(range(1, epochs + 1), desc="Epochs", leave=True, bar_format="{l_bar}{bar}| Epoch {n_fmt}/{total_fmt}")
    for epoch in epoch_pbar:
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = eval_step(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Update epoch progress bar with metrics.
        epoch_pbar.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "Val Acc": f"{val_acc:.4f}"
        })

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping based on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

        # Gradual unfreezing after a specific epoch.
        if epoch == unfreeze_epoch:
            if hasattr(model, "blocks"):
                if unfreeze_option == "last":
                    print("\nUnfreezing the last block for fine-tuning...")
                    for block in model.blocks[-1:]:
                        for param in block.parameters():
                            param.requires_grad = True
                elif unfreeze_option == "last_two":
                    print("\nUnfreezing the last two blocks for fine-tuning...")
                    for block in model.blocks[-2:]:
                        for param in block.parameters():
                            param.requires_grad = True
                # Additional unfreezing options can be added here.
                # Adjust the learning rate for newly unfrozen parameters.
                for g in optimizer.param_groups:
                    g["lr"] = base_lr * 0.5
            else:
                print("\nModel does not have a 'blocks' attribute; manual unfreezing may be required.")

        if scheduler is not None:
            scheduler.step()

    return history

if __name__ == "__main__":
    # Example usage (this block is optional and for testing purposes):
    # from going_modular.custom_dataset import CustomImageDataset
    # from torch.utils.data import DataLoader
    # import timm
    # You can initialize a model, create dummy dataloaders, and call the train_model function here.
    pass
