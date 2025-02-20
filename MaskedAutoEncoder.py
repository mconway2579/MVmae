import torch
import torch.nn as nn
import math

class MaskedAutoEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, nhead=4, mask_ratio=0.5, max_seq_length=50176):
        super(MaskedAutoEncoder, self).__init__()
        self.mask_ratio = mask_ratio
        self.max_seq_length = max_seq_length+1 # set maximum sequence length as needed

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Project input to hidden dimension.
        self.embedding = nn.Linear(3, hidden_dim)

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
        self.output_layer = nn.Linear(hidden_dim, 3)

        self.to(self.device)

    def forward(self, x, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        #print(f"\n\nin forward")
        # Create one mask value per pixel (broadcasted over channels)
        mask = (torch.rand(x.size(0), 1, x.size(2), x.size(3), device=self.device) > mask_ratio).float()
        x = x * mask
        #print(f"original{x.shape=}")
        x = x.permute(0,2,3,1)
        #print(f"{masked_x.shape=}")

        #print(f"permuted {x.shape=}")
        dims = x.shape
        x = x.reshape(dims[0], dims[1]*dims[2], dims[3])
        #print(f"flattened {x.shape=}")

        # Create a random binary mask based on the mask_ratio.
        

        x_embedded = self.embedding(x)
        #print(f"after embedded {x_embedded.shape=}")
        x_embedded = x_embedded + self.positional_encoding[:x_embedded.size(1), :].unsqueeze(0)
        #print(f"after PE {x_embedded.shape=}")

        # Add class token
        batch_size = x_embedded.size(0)
        #print(f"{batch_size=}")
        class_token = self.class_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        #print(f"{class_token.shape=}")

        x_embedded = torch.cat((class_token, x_embedded), dim=1)
        #print(f"after CLS token {x_embedded.shape=}")


        # Apply the transformer encoder.
        memory = self.transformer_encoder(x_embedded) 

        # Apply the transformer decoder.
        decoded = self.transformer_decoder(x_embedded, memory)

        # Extract the class token from the decoder output
        class_token_output = decoded[:, 0, :]
        #print(f"{class_token_output.shape=}")
        # Remove the sequence dimension and class token.

        # Project back to input dimension.
        reconstructed = self.output_layer(decoded[:, 1:, :])
        #print(f"{reconstructed.shape=}")

        reconstructed = reconstructed.reshape(dims[0], dims[1], dims[2], dims[3])
        reconstructed = reconstructed.permute(0,3,1,2)
        return reconstructed, mask, class_token_output

if __name__ == "__main__":
    from MessyTableInterface import MessyTableDataset, display_batch
    from torch.utils.data import DataLoader
    import os
    hidden_dim = 128
    batch_size = 4
     
    file_path = './MessyTableData/labels/train.json'

    dataset = MessyTableDataset(file_path, set_size=1, img_shape=(32,32))
    seq_length = dataset.image_shape[0] * dataset.image_shape[1]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    data = next(iter(data_loader))
    _, _, object_imgs_batch, _, _ = data
    print(f"{len(object_imgs_batch)=}")
    print(f"{object_imgs_batch[0].shape=}")
    object_imgs_batch = object_imgs_batch[0]

    model = MaskedAutoEncoder(hidden_dim, max_seq_length=seq_length)
    
    output, mask, emb = model(object_imgs_batch)

    print("Reconstructed Output:", output.shape)
    print("Mask:", mask.shape)
    print(f"{emb.shape=}")