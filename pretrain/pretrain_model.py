import torch
import torch.nn as nn

class CNN_Transformer(nn.Module):
    def __init__(self, transformer_emb_size=1024, num_heads=8, dim_feedforward=2048, num_layers=8, dropout=0.1):
        super().__init__()

        self.conv=nn.Conv1d(56, 56, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.linear = nn.Linear(419, transformer_emb_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_emb_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x, input_masks_invert):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.linear(x)
        x = self.transformer_encoder(x, src_key_padding_mask=input_masks_invert)
        return x
    
class CNN_Decoder(nn.Module):
    def __init__(self, transformer_emb_size=1024):
        super().__init__()
        self.linear = nn.Linear(transformer_emb_size, 838)
        self.deconv = nn.ConvTranspose1d(56, 56, kernel_size=3)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.linear(x)
        x = self.deconv(x)
        x = self.relu(x)
        return x

class EEGAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_transformer = CNN_Transformer()
        self.decoder = CNN_Decoder()

    def forward(self, x, input_masks_invert):
        x = self.cnn_transformer(x, input_masks_invert)
        x = self.decoder(x)
        return x

class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers):
        super().__init__()
        self.cnn_transformer = CNN_Transformer()
        self.llm = pretrained_layers
    
    def forward(self, x, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        embedding = self.cnn_transformer(x, input_masks_invert)
        output = self.llm(inputs_embeds=embedding, attention_mask=input_masks_batch, return_dict=True, labels=target_ids_batch_converted)
        return output



# class CNN_Transformer(nn.Module):
#     def __init__(self, cnn_emb_size=40, transformer_emb_size=1024, num_heads=8, num_layers=6, dropout=0.1):
#         super().__init__()

#         self.cnn_encoder = nn.Sequential(
#             nn.Conv2d(1, 40, (1, 41), (1, 1)),
#             nn.Conv2d(40, 40, (105, 1), (1, 1)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.AvgPool2d((1, 75), (1, 15)),
#             nn.Dropout(0.5),
#         )
        
#         self.flatten = nn.Flatten(start_dim=2)  # 将卷积输出的高维度展平为一个序列
#         self.linear = nn.Linear(660, transformer_emb_size)  # 将展平后的序列映射到Transformer的嵌入维度
        
#         # 定义Transformer层
#         encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_emb_size, nhead=num_heads, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.cnn_encoder(x)
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.transformer_encoder(x)

#         return x


# class CNNDecoder(nn.Module):
#     def __init__(self, cnn_emb_size=40, transformer_emb_size=1024):
#         super().__init__()

#         self.linear = nn.Linear(transformer_emb_size, 660)
#         self.unflatten = nn.Unflatten(dim=2, unflattened_size=(1, 660))
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(40, 40, (1, 75), (1, 15)),
#             nn.ELU(),
#             nn.BatchNorm2d(40),
#             nn.ConvTranspose2d(40, 40, (105, 1), (1, 1)),
#             nn.ELU(),
#             nn.BatchNorm2d(40),
#             nn.ConvTranspose2d(40, 1, (1, 41), (1, 1)),
#             nn.ELU(),
#         )
    
#     def forward(self, x):
#         x = self.linear(x)
#         x = self.unflatten(x)
#         x = self.decoder(x)
#         return x


# class EEGAutoencoder(nn.Module):
#     def __init__(self, cnn_emb_size=40, transformer_emb_size=1024, num_heads=8, num_layers=6, dropout=0.1):
#         super().__init__()

#         self.encoder=CNN_Transformer()
#         self.decoder = CNNDecoder()
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = x.squeeze(1)
#         return x