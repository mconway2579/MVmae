from MessyTableInterface import MessyTableDataset
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
torch.set_float32_matmul_precision('high')
from torch import compile
from torch.amp import GradScaler


def get_output_imgs(model, data_loader, save_file):
    model.eval()
    for object_imgs_batch, label_batch in data_loader:
        #print(data)
        #print(type(data))
        #print(len(data))
        B, S, C, H, W = object_imgs_batch.shape
        object_imgs_batch= object_imgs_batch.reshape(-1, C, H, W).to(model.device, non_blocking=True)        

        fig_side_length = int(np.ceil(np.sqrt(len(object_imgs_batch))))
        #print(f"{fig_side_length=}")
        fig, axes = plt.subplots(fig_side_length, fig_side_length)
        axes = axes.flatten()

        output, masked_imgs, emb = model(object_imgs_batch)
        for i, (img, pred, masked_img) in enumerate(zip(object_imgs_batch, output, masked_imgs)):
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


def pre_train_epoch_func(model, loader, name, optimizer=None, scaler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    total_loss = 0
    n = 0
    for object_imgs_batch, label_batch in tqdm(loader, desc=name):
        B, S, C, H, W = object_imgs_batch.shape
        object_imgs_batch= object_imgs_batch.reshape(-1, C, H, W).to(model.device, non_blocking=True)
        #print(f"{object_imgs_batch.shape=}")
        n += len(object_imgs_batch)
        output = None
        mask = None
        emb = None
        loss = None
        with torch.autocast(device_type="cuda"):
            if optimizer is None:
                with torch.inference_mode():
                    with torch.no_grad():
                        #print("before inference")
                        output, mask, emb = model(object_imgs_batch)
                        #print("After inference")

            else:
                output, mask, emb = model(object_imgs_batch)

            loss = F.mse_loss(output, object_imgs_batch)
            total_loss += loss.item()

        if optimizer is not None:
            #print("before optimization")
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #print("after optimization")

    return total_loss / n


def pretrain(model, train_loader, val_loaders, test_loaders, n_epochs, save_dir, img_dir):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    best_model = None
    training_loss_history = []
    validation_loss_history = []
    scaler = GradScaler('cuda')
    print(f"training on {model.device}")

    for epoch in range(n_epochs):
        print(f"\n\nEpoch {epoch+1}/{n_epochs}:")
        #do a training epoch
        train_loss = pre_train_epoch_func(model, train_loader, name="Training", optimizer=optimizer, scaler=scaler)
        training_loss_history.append(train_loss)
        print(f"Train Loss: {train_loss}")

        #do a validation epoch for each validation set
        validation_losses = [(suffix, pre_train_epoch_func(model, val_loader, name=suffix)) for suffix, val_loader in val_loaders]
        validation_loss_history.append(validation_losses)
        print(f"Validation Losses: {validation_losses}")

        #if this is our best model, save it
        total_val_loss = [loss for name, loss in validation_losses if name == "val.json"][0]
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model = model.state_dict()
            torch.save(best_model, f"{save_dir}/best_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss}")

        #save the output images for the test set
        epoch_dir = f"{img_dir}/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        get_output_imgs(model, train_loader, f"{epoch_dir}/train.png")
        get_output_imgs(model, val_loaders[-1][1], f"{epoch_dir}/val.png")
        get_output_imgs(model, test_loaders[-1][1], f"{epoch_dir}/test.png")
        print()
    
    model.load_state_dict(best_model)
    test_losses = [(name, pre_train_epoch_func(model, test_loader, name = name)) for name, test_loader in test_loaders]
    total_test_loss = [loss for name, loss in test_losses if name == "test.json"][0]
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


if __name__ == "__main__":
    hidden_dim = 128
    batch_size = 8
    set_size = 4
    patch_size=128
    n_pretrain_epochs = 2
    mask_percentage=0.75
    loader_workers = 16
    out_dir = f"./examples/pretraining/"
    os.makedirs(out_dir, exist_ok=True)
    img_dir = f"{out_dir}/imgs"
    os.makedirs(img_dir, exist_ok=True)
    model = MaskedAutoEncoder(hidden_dim, max_seq_length=1024**2, mask_ratio=mask_percentage)
    if torch.cuda.is_available():
        model = model.to('cuda')
    train_dataset = MessyTableDataset("./MessyTableData/labels/train.json", set_size=set_size, train=True)
    val_dataset = MessyTableDataset("./MessyTableData/labels/val.json", set_size=set_size, train=False)
    test_dataset = MessyTableDataset("./MessyTableData/labels/test.json", set_size=set_size, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True)
    val_laoder = [("val.json", DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers, pin_memory=True))]
    test_loader = [("test.json", DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers, pin_memory=True))]
    model = pretrain(model, train_loader, val_laoder, test_loader, n_pretrain_epochs, out_dir, img_dir)