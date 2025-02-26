import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from config import IMG_SHAPE
class MaskedAutoEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, nhead=4, mask_ratio=0.5, patch_size=32):
        super(MaskedAutoEncoder, self).__init__()
        self.mask_ratio = mask_ratio
        self.max_seq_length = (IMG_SHAPE[0] * IMG_SHAPE[1]) + 1 # set maximum sequence length as needed

        self.patch_size = patch_size
        self.pixels_per_patch = patch_size * patch_size *3

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Project input to hidden dimension.
        self.embedding = nn.Linear(self.pixels_per_patch, hidden_dim)

        #initialize sinisoidal positional encoding
        pe = torch.zeros(self.max_seq_length, hidden_dim)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)

        #Create class token for embedding vectors
        self.class_token = nn.Parameter(torch.zeros(hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to project back to input dimension.
        self.output_layer = nn.Linear(hidden_dim, self.pixels_per_patch)

        self.to(self.device)

    def forward(self, x, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        #print(f"\n\nin forward")

        #print(f"original{x.shape=}")
        B, C, H, W = x.shape
        #print(f"{B=}, {C=}, {H=}, {W=}")
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        #print(f"{num_patches=}, {self.pixels_per_patch=}")
        # Create one mask value per pixel (broadcasted over channels)
        


        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        #print(f"after unfold {x.shape=}")
        #num_patches = x.shape[-1]
        x = x.transpose(2,1)
        #print(f"after transpose {x.shape=}")

        mask = (torch.rand(x.size(0), x.size(1),1, device=self.device) > mask_ratio).float()
        #print(f"{mask.shape=}")
        x = x * mask
        masked_seq = x
        #print(f"masked x = {x.shape}")

        #print(f"before embedded {x.shape=}")
        x = self.embedding(x)
        #print(f"after embedded {x.shape=}")
        x += self.positional_encoding[:x.size(1), :].unsqueeze(0)
        #print(f"after PE {x.shape=}")

        # Add class token
        batch_size = x.size(0)
        #print(f"{batch_size=}")
        class_token = self.class_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        #print(f"{class_token.shape=}")

        x = torch.cat((class_token, x), dim=1)
        #print(f"after CLS token {x.shape=}")


        # Apply the transformer encoder.
        memory = self.transformer_encoder(x) 

        # Apply the transformer decoder.
        decoded = self.transformer_decoder(x, memory)

        # Extract the class token from the decoder output
        class_token_output = decoded[:, 0, :]
        #print(f"{class_token_output.shape=}")
        # Remove the sequence dimension and class token.

        # Project back to input dimension.
        reconstructed_sequence = self.output_layer(decoded[:, 1:, :])
        #print(f"{reconstructed.shape=}")

        
        reconstructed_sequence = reconstructed_sequence.transpose(2,1)
        #print(f"after transpose {x.shape=}")
        reconstructed_sequence = F.fold(reconstructed_sequence, (H, W), kernel_size=self.patch_size, stride=self.patch_size)

        masked_img = masked_seq.transpose(2,1)
        masked_img = F.fold(masked_img, (H, W), kernel_size=self.patch_size, stride=self.patch_size)

        return reconstructed_sequence, masked_img, class_token_output
if __name__ == "__main__":
    from MessyTableInterface import MessyTableDataset, display_batch
    from torch.utils.data import DataLoader
    import os
    import matplotlib.pyplot as plt
    hidden_dim = 256
    batch_size = 4
    patch_size = 16
     
    file_path = './MessyTableData/labels/train.json'

    dataset = MessyTableDataset(file_path, set_size=1, train=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #display_batch(data_loader)
    object_imgs_batch, instance_scene_batch = next(iter(data_loader))
    B, S, C, H, W = object_imgs_batch.shape
    object_imgs_batch= object_imgs_batch.reshape(-1, C, H, W)
    print(f"{object_imgs_batch.shape=}, {len(instance_scene_batch)=}")

    model = MaskedAutoEncoder(hidden_dim, patch_size=patch_size)
    object_imgs_batch = object_imgs_batch.to(model.device)
    
    output, masked_imgs, emb = model(object_imgs_batch)
    print(f"{emb.shape=}")

    print(f"Reconstructed Output: {output.shape}")
    print(f"masked image: {masked_imgs.shape}")
    print(f"{output.shape=}, {object_imgs_batch.shape=}")

    output = output.permute(0,2,3,1)
    object_imgs_batch = object_imgs_batch.permute(0,2,3,1)
    masked_imgs = masked_imgs.permute(0,2,3,1)

    print(f"{output.shape=}, {object_imgs_batch.shape=}, {masked_imgs.shape=}")
    for original, masked_img, reconstructed in zip(object_imgs_batch, masked_imgs, output):
        print(f"{original.shape=}, {reconstructed.shape=}, {masked_imgs.shape=}")
        display_img = torch.cat((original, masked_img,reconstructed), dim=1).detach().cpu().numpy()
        plt.imshow(display_img)
        plt.show()
    #print("Mask:", mask.shape)
    #print(f"{emb.shape=}")