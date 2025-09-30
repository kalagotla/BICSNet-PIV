import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_layers=2):
        super(Encoder, self).__init__()
        layers = []
        in_channels = 3
        out_channels = 64
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DualEncoder(nn.Module):
    def __init__(self, num_layers=2):
        super(DualEncoder, self).__init__()
        layers = []
        in_channels = 6
        out_channels = 64
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.layers(x)


class ScalarEmbedding(nn.Module):
    def __init__(self, scalar_input_dim, embed_dim=128):
        super(ScalarEmbedding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(scalar_input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, scalars):
        return self.fc(scalars)


class Decoder(nn.Module):
    def __init__(self, num_layers=3, enc_out_channels=128, dual_enc_out_channels=128, scalar_embed_dim=128,
                 use_scalars=True):
        super(Decoder, self).__init__()
        self.use_scalars = use_scalars
        in_channels = 3 + enc_out_channels + dual_enc_out_channels
        if use_scalars:
            in_channels += scalar_embed_dim  # add scalar embedding channels if using scalars

        layers = []
        out_channels = 128
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels //= 2
        layers.append(nn.Conv2d(in_channels, 3, 3, padding=1))  # final output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x, x_enc, dual_enc, scalar_embed=None):
        if self.use_scalars and scalar_embed is not None:
            # Expand scalar embedding spatially to match image dimensions and concatenate
            scalar_embed_expanded = scalar_embed.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat((x, x_enc, dual_enc, scalar_embed_expanded), dim=1)
        else:
            x = torch.cat((x, x_enc, dual_enc), dim=1)
        return self.layers(x)


class Net(nn.Module):
    def __init__(self, num_layers_encoder=2, num_layers_decoder=3, scalar_input_dim=2, scalar_embed_dim=128,
                 use_scalars=True):
        super(Net, self).__init__()
        self.use_scalars = use_scalars
        self.encoder1 = Encoder(num_layers_encoder)
        self.encoder2 = Encoder(num_layers_encoder)
        self.dual_encoder = DualEncoder(num_layers_encoder)

        if use_scalars:
            self.scalar_embed = ScalarEmbedding(scalar_input_dim, scalar_embed_dim)

        # Calculate channels for Decoder
        encoder_out_channels = 64 * (2 ** (num_layers_encoder - 1))
        dual_encoder_out_channels = 64 * (2 ** (num_layers_encoder - 1))

        self.decoder1 = Decoder(num_layers_decoder, encoder_out_channels, dual_encoder_out_channels, scalar_embed_dim,
                                use_scalars)
        self.decoder2 = Decoder(num_layers_decoder, encoder_out_channels, dual_encoder_out_channels, scalar_embed_dim,
                                use_scalars)

    def forward(self, img1, img2, scalars=None):
        # Encode images
        enc1_out = self.encoder1(img1)
        enc2_out = self.encoder2(img2)
        dual_enc_out = self.dual_encoder(img1, img2)

        if self.use_scalars and scalars is not None:
            # Embed scalar inputs if they are provided
            scalar_embed = self.scalar_embed(scalars)
        else:
            scalar_embed = None

        # Decode with or without scalar embeddings
        dec1_out = self.decoder1(img1, enc1_out, dual_enc_out, scalar_embed)
        dec2_out = self.decoder2(img2, enc2_out, dual_enc_out, scalar_embed)
        return dec1_out, dec2_out