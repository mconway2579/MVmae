from MessyTableInterface import MessyTableDataset
from torch.utils.data import DataLoader
from MaskedAutoEncoder import MaskedAutoEncoder
import os
os.environ["PYTORCH_FX_DISABLE_SYMBOLIC_SHAPES"] = "1"
import warnings
warnings.filterwarnings("ignore", message=".*not in var_ranges.*")
import torch
print(f"{torch.__version__=}")  # PyTorch version
print(f"{torch.version.cuda=}")  # CUDA version
torch.set_float32_matmul_precision('high')
from torch.profiler import profile, record_function, ProfilerActivity
from pre_training import pretrain
from association_training import train_association



def run_experiment(hidden_dim = 128, batch_size = 64, set_size = 3, patch_size=32, n_pretrain_epochs = 5, n_association_epochs=5, mask_percentage=0.75, loader_workers = 16):
    #save directories
    os.makedirs("./outputs/", exist_ok=True)

    out_dir = f"./outputs/{hidden_dim=}_{batch_size=}_{patch_size=}_{n_pretrain_epochs=}_{n_association_epochs}_{mask_percentage=}"
    os.makedirs(out_dir, exist_ok=True)
    
    pretrain_dir = f"{out_dir}/pretrain"
    os.makedirs(pretrain_dir, exist_ok=True)

    pretrain_img_dir = f"{pretrain_dir}/imgs"
    os.makedirs(pretrain_img_dir, exist_ok=True)

    association_dir = f"{out_dir}/association"
    os.makedirs(association_dir, exist_ok=True)

    assocition_img_dir = f"{association_dir}/imgs"
    os.makedirs(assocition_img_dir, exist_ok=True)
    

    #Datasets and data loaders
    train_dataset = MessyTableDataset("./MessyTableData/labels/train.json", set_size=set_size, train=True)
    test_datasets = []
    val_datasets = []
    #
    #for suffix in ["_hard.json", ".json"]:
    #for suffix in ["_easy.json", "_medium.json", "_hard.json", ".json"]:
    for suffix in [".json"]:
        test_datasets.append((f"test{suffix}", MessyTableDataset(f"./MessyTableData/labels/test{suffix}", set_size=set_size, train=False)))
        val_datasets.append((f"val{suffix}", MessyTableDataset(f"./MessyTableData/labels/val{suffix}", set_size=set_size, train=False)))

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True)
    test_loaders = [(name, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers, pin_memory=True)) for name, dataset in test_datasets]
    val_loaders = [(name, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True)) for name, dataset in val_datasets]

    #create model, optimizer, and loss function
    model = MaskedAutoEncoder(hidden_dim, mask_ratio=mask_percentage, patch_size=patch_size)
    #model = compile(model)  # Wrap the model with torch.compile

    # Move model to CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Pretrain the model
    if os.path.exists(f"{pretrain_dir}/best_model.pth"):
        model.load_state_dict(torch.load(f"{pretrain_dir}/best_model.pth", weights_only=True))
        print(f"Loaded pretrain model from {pretrain_dir}/best_model.pth")
    else:
        model = pretrain(model, train_loader, val_loaders, test_loaders, n_pretrain_epochs, pretrain_dir, pretrain_img_dir)
    #train association
    model = train_association(model, train_loader, val_loaders, test_loaders, n_association_epochs, association_dir, assocition_img_dir)


if __name__ == "__main__":
    experiment_configs = [
        #{"hidden_dim": 128, "batch_size": 32, "set_size": 1, "n_epochs": 5, "mask_percentage": 0},
        #{"hidden_dim": 128, "batch_size": 32, "set_size": 1, "n_epochs": 5, "mask_percentage": 0.25},
        #{"hidden_dim": 128, "batch_size": 32, "set_size": 1, "n_epochs": 5, "mask_percentage": 0.5},
        #{"hidden_dim": 128, "batch_size": 16, "set_size": 1, "n_epochs": 5, "mask_percentage": 0.75},
        {"hidden_dim": 128, "batch_size": 8, "set_size": 4, "n_pretrain_epochs": 5, "n_association_epochs":5, "mask_percentage": 0.9},
        #{"hidden_dim": 128, "batch_size": 32, "set_size": 1, "n_epochs": 5, "mask_percentage": 1},
    ]

    for config in experiment_configs:
        run_experiment(**config)


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