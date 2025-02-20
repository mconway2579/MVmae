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
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
torch.set_float32_matmul_precision('high')
from torch import compile
from torch.amp import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity


#@torch.compile
def epoch_func(model, loader, loss_fn, name, optimizer=None, scaler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    total_loss = 0
    n = 0
    for data in tqdm(loader, desc=name):
        object_imgs_batch= data[2][0].to(model.device, non_blocking=True)
        n += len(object_imgs_batch)
        output = None
        mask = None
        emb = None
        loss = None
        with torch.autocast(device_type="cuda"):
            if optimizer is None:
                with torch.inference_mode():
                    with torch.no_grad():
                        output, mask, emb = model(object_imgs_batch)
            else:
                output, mask, emb = model(object_imgs_batch)

            loss = loss_fn(output, object_imgs_batch)
            total_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return total_loss / n

def get_output_imgs(model, data_loader, save_file):
    model.eval()
    for data in data_loader:
        #print(data)
        #print(type(data))
        #print(len(data))
        object_imgs_batch= data[2][0].to(model.device, non_blocking=True)
        #print(f"{type(object_imgs_batch)=}")
        #print(f"{object_imgs_batch.shape=}")
        

        fig_side_length = int(np.ceil(np.sqrt(len(object_imgs_batch))))
        #print(f"{fig_side_length=}")
        fig, axes = plt.subplots(fig_side_length, fig_side_length)
        axes = axes.flatten()

        output, mask, emb = model(object_imgs_batch)
        for i, (img, pred, m) in enumerate(zip(object_imgs_batch, output, mask)):
            masked_img = img * m
            masked_img = masked_img.permute(1,2,0)
            img = img.permute(1,2,0)
            pred = pred.permute(1,2,0)
            new_img = torch.cat((img, masked_img, pred), dim=1)
            new_img = new_img.cpu().detach().numpy()
            axes[i].imshow(new_img)
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(save_file)
        return save_file

def run_experiment(hidden_dim = 128, batch_size = 64, set_size = 1, img_shape = (64, 64), n_epochs = 5, mask_percentage=0.75, loader_workers = 30):
    #save directories
    os.makedirs("./outputs/", exist_ok=True)
    out_dir = f"./outputs/{hidden_dim=}_{batch_size=}_{img_shape=}_{n_epochs=}_{mask_percentage=}"
    os.makedirs(out_dir, exist_ok=True)
    img_dir = f"{out_dir}/imgs"
    os.makedirs(img_dir, exist_ok=True)

    #Datasets and data loaders
    train_dataset = MessyTableDataset("./MessyTableData/labels/train.json", set_size=set_size, img_shape=img_shape, train=True)
    test_datasets = []
    val_datasets = []
    #
    #for suffix in ["_hard.json", ".json"]:
    #for suffix in [".json"]:
    for suffix in ["_easy.json", "_medium.json", "_hard.json", ".json"]:
        test_datasets.append((f"test{suffix}", MessyTableDataset(f"./MessyTableData/labels/test{suffix}", set_size=set_size, img_shape=img_shape, train=False)))
        val_datasets.append((f"val{suffix}", MessyTableDataset(f"./MessyTableData/labels/val{suffix}", set_size=set_size, img_shape=img_shape, train=False)))

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True)
    test_loaders = [(name, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers, pin_memory=True)) for name, dataset in test_datasets]
    val_loaders = [(name, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True)) for name, dataset in val_datasets]

    #create model, optimizer, and loss function
    seq_length = train_dataset.image_shape[0] * train_dataset.image_shape[1]
    model = MaskedAutoEncoder(hidden_dim, max_seq_length=seq_length, mask_ratio=mask_percentage)
    model = compile(model)  # Wrap the model with torch.compile

    # Move model to CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    #create variables for saving and logging
    best_val_loss = float("inf")
    best_model = None
    training_loss_history = []
    validation_loss_history = []
    scaler = GradScaler('cuda')

    #training loop
    print(f"training on {model.device}")
    for epoch in range(n_epochs):
        print(f"\n\nEpoch {epoch+1}/{n_epochs}:")
        #do a training epoch
        train_loss = epoch_func(model, train_loader, mse_loss, name="Training", optimizer=optimizer, scaler=scaler)
        training_loss_history.append(train_loss)
        print(f"Train Loss: {train_loss}")

        #do a validation epoch for each validation set
        validation_losses = [(suffix, epoch_func(model, val_loader, mse_loss, name=suffix)) for suffix, val_loader in val_loaders]
        validation_loss_history.append(validation_losses)
        print(f"Validation Losses: {validation_losses}")

        #if this is our best model, save it
        total_val_loss = [loss for name, loss in validation_losses if name == "val.json"][0]
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model = model.state_dict()
            torch.save(best_model, f"{out_dir}/best_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss}")

        #save the output images for the test set
        epoch_dir = f"{img_dir}/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        output_imgs = get_output_imgs(model, train_loader, f"{epoch_dir}/train.png")
        output_imgs = get_output_imgs(model, val_loaders[-1][1], f"{epoch_dir}/val.png")
        output_imgs = get_output_imgs(model, test_loaders[-1][1], f"{epoch_dir}/test.png")
        print()
    
    model.load_state_dict(best_model)
    test_losses = [(name, epoch_func(model, test_loader, mse_loss, name = name)) for name, test_loader in test_loaders]
    total_test_loss = [loss for name, loss in test_losses if name == "test.json"][0]
    print(f"Test Loss: {total_test_loss}")
    with open(f"{out_dir}/metrics.txt", "w") as f:
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
    plt.savefig(f"{out_dir}/losses.png")
    print(f"Results saved to {out_dir}")
    print("Done")


if __name__ == "__main__":
    experiment_configs = [
        {"hidden_dim": 128, "batch_size": 64, "set_size": 1, "img_shape": (64, 64), "n_epochs": 5, "mask_percentage": 0},
        {"hidden_dim": 128, "batch_size": 64, "set_size": 1, "img_shape": (64, 64), "n_epochs": 5, "mask_percentage": 0.25},
        {"hidden_dim": 128, "batch_size": 64, "set_size": 1, "img_shape": (64, 64), "n_epochs": 5, "mask_percentage": 0.5},
        {"hidden_dim": 128, "batch_size": 64, "set_size": 1, "img_shape": (64, 64), "n_epochs": 5, "mask_percentage": 0.75},
        {"hidden_dim": 128, "batch_size": 64, "set_size": 1, "img_shape": (64, 64), "n_epochs": 5, "mask_percentage": 0.9},
        {"hidden_dim": 128, "batch_size": 64, "set_size": 1, "img_shape": (64, 64), "n_epochs": 5, "mask_percentage": 1},
    ]

    for config in experiment_configs:
        run_experiment(**config)
    run_experiment()


    
"""
if __name__ == "__main__":
    os.makedirs("./profile/", exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,       # Disable shape recording if not essential
        profile_memory=False,      # Disable memory profiling to reduce overhead
        with_stack=False           # Disable stack tracing to lower data collection cost
    ) as prof:
        with record_function("run_experiment"):
            run_experiment()
    #Export the complete trace to a JSON file for viewing in Chrome Trace Viewer
    prof.export_chrome_trace("./profile/profile_trace.json")
    
    # Optionally, write a summary table to a text file
    summary = prof.key_averages().table(sort_by="cuda_time_total")
    with open("./profile/profile_summary.txt", "w") as f:
        f.write(summary)
"""