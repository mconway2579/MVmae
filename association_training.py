from MessyTableInterface import MessyTableDataset, display_batch
from torch.utils.data import DataLoader
from MaskedAutoEncoder import MaskedAutoEncoder
from tqdm import tqdm
import os
os.environ["PYTORCH_FX_DISABLE_SYMBOLIC_SHAPES"] = "1"
import warnings
warnings.filterwarnings("ignore", message=".*not in var_ranges.*")
import torch
print(f"{torch.__version__=}")  # PyTorch version
print(f"{torch.version.cuda=}")  # CUDA version
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
torch.set_float32_matmul_precision('high')
from torch import compile
from torch.amp import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity

def association_loss_func(emb, labels):
    #print(f"{emb.shape=}")
    #print(f"{labels.shape=}")
    matches_target_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).all(dim=-1)
    distance_target_matrix = (~matches_target_matrix).float() * 2

    #print(f"{target_matrix.shape=}")
    #num_ones = matches_target_matrix.sum()
    #print("Number of ones in the equivalence matrix:", num_ones.item())
    cos_dist_matrix = 1-F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    #print(f"{cos_dist_matrix.shape=}")
    #print(f"{distance_target_matrix.shape=}")
    return F.mse_loss(cos_dist_matrix, distance_target_matrix)
def association_epoch_func(model, loader, name, optimizer=None, scaler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    total_loss = 0
    acc_association_loss=0
    acc_reconstruction_loss=0
    n = 0
    for object_imgs_batch, label_batch in tqdm(loader, desc=name):
        B, S, C, H, W = object_imgs_batch.shape
        object_imgs_batch= object_imgs_batch.reshape(-1, C, H, W).to(model.device, non_blocking=True)
        label_batch= label_batch.reshape(-1, 22).to(model.device, non_blocking=True)

        #print(f"{object_imgs_batch.shape=}")
        n += len(object_imgs_batch)
        output = None
        masks = None
        embeddings = None
        loss = None
        with torch.autocast(device_type="cuda"):
            if optimizer is None:
                with torch.inference_mode():
                    with torch.no_grad():
                        #print("before inference")
                        output, mask, emb = model(object_imgs_batch, mask_ratio=0)
                        #print("After inference")

            else:
                output, mask, emb = model(object_imgs_batch, mask_ratio=0)

            reconstruction_loss = F.mse_loss(output, object_imgs_batch)
            association_loss = association_loss_func(emb, label_batch)
            loss = reconstruction_loss + association_loss 
            total_loss += loss.item()
            acc_association_loss = association_loss.item()
            acc_reconstruction_loss = reconstruction_loss.item()

        if optimizer is not None:
            #print("before optimization")
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #print("after optimization")

    return {"total_loss":total_loss / n, "association_loss":acc_association_loss/n, "reconstruction_loss":acc_reconstruction_loss/n}

def train_association(model, train_loader, val_loaders, test_loaders, n_epochs, save_dir, img_dir):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_model = None
    training_loss_history = []
    validation_loss_history = []
    scaler = GradScaler('cuda')
    print(f"training on {model.device}")
    output_imgs = get_output_imgs(model, test_loaders[-1][1], f"{save_dir}/tpre_assocation_training.png")
    for epoch in range(n_epochs):
        print(f"\n\nEpoch {epoch+1}/{n_epochs}:")
        #do a training epoch
        train_loss = association_epoch_func(model, train_loader, name="Training", optimizer=optimizer, scaler=scaler)
        training_loss_history.append(train_loss)
        print(f"Train Loss: {train_loss}")

        #do a validation epoch for each validation set
        validation_losses = [(suffix, association_epoch_func(model, val_loader, name=suffix)) for suffix, val_loader in val_loaders]
        validation_loss_history.append(validation_losses)
        print(f"Validation Losses: {validation_losses}")

        #if this is our best model, save it
        total_val_loss = [loss for name, loss in validation_losses if name == "val.json"][0]["total_loss"]
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model = model.state_dict()
            torch.save(best_model, f"{save_dir}/best_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss}")

        #save the output images for the test set
        epoch_dir = f"{img_dir}/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        output_imgs = get_output_imgs(model, train_loader, f"{epoch_dir}/train.png")
        output_imgs = get_output_imgs(model, val_loaders[-1][1], f"{epoch_dir}/val.png")
        output_imgs = get_output_imgs(model, test_loaders[-1][1], f"{epoch_dir}/test.png")
        print()
    
    model.load_state_dict(best_model)
    test_losses = [(name, association_epoch_func(model, test_loader, name = name)) for name, test_loader in test_loaders]
    total_test_loss = [loss for name, loss in test_losses if name == "test.json"][0]["total_loss"]
    print(f"Test Loss: {total_test_loss}")
    with open(f"{save_dir}/metrics.txt", "w") as f:
        f.write(f"Test loss: {total_test_loss}\n")
        f.write(f"Best Model Validation Loss: {best_val_loss}\n")
        
        f.write("Test Losses:\n")
        for name, loss in test_losses:
            f.write(f"   {name}: {loss}\n")

        for i, (train_loss, val_losses) in enumerate(zip(training_loss_history, validation_loss_history)):
            f.write(f"Epoch {i}:\n")
            f.write(f"   Train Loss: {train_loss}\n")
            f.write(f"   Validation Losses:\n")
            for name, loss in val_losses:
                f.write(f"      {name}: {loss}\n")
            f.write("\n")

    fig, ax = plt.subplots()
    ax.plot(training_loss_history, label="Train Loss")


    val_losses_dict = {name: [] for name, _ in val_loaders}
    for val_losses in validation_loss_history:
        for name, loss in val_losses:
            val_losses_dict[name].append(loss)
    for name, losses in val_losses_dict.items():
        ax.plot(losses, label=f"{name} Loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Losses")
    plt.savefig(f"{save_dir}/losses.png")
    print(f"Results saved to {save_dir}")
    print("Done")
    return model
