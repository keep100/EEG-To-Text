import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F




""" main architecture for open vocabulary EEG-To-Text decoding"""
class BrainTranslator(nn.Module):
    def __init__(
        self,
        pretrained_layers,
        in_feature=840,
        decoder_embedding_size=1024,
        additional_encoder_nhead=8,
        additional_encoder_dim_feedforward=2048,
    ):
        super(BrainTranslator, self).__init__()
        model = nn.Sequential()
        model.add_module(
            "gru",
            nn.GRU(
                in_feature,
                hidden_size=512,
                batch_first=True,
                bidirectional=True,
            ),
        )
        # model.add_module("pe",PositionalEncoding(d_model=in_feature,dropout_prob=0.1))
        model.add_module(
            "mhte",
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=in_feature,
                    nhead=additional_encoder_nhead,
                    dim_feedforward=additional_encoder_dim_feedforward,
                    batch_first=True,
                ),
                num_layers=10,
            ),
        )
        model.add_module("vq", Quantize())
        model.add_module("fc1", nn.Linear(512 * 2, in_feature))
        model.add_module("fc2", nn.Linear(in_feature, decoder_embedding_size))
        model.add_module("norm1", nn.LayerNorm(in_feature, eps=1e-5))
        model.add_module("norm2", nn.LayerNorm(decoder_embedding_size, eps=1e-5))
        model.add_module("llm", pretrained_layers)
        self.model = model

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        gru_output, _ = self.model.gru(input_embeddings_batch)
        norm1_output = F.relu(self.model.norm1(self.model.fc1(gru_output)))

        # position_embedding=self.model.pe(norm1_output)
        # use src_key_padding_masks
        encoded_embedding = self.model.mhte(
            norm1_output, src_key_padding_mask=input_masks_invert
        )
        norm2_output = F.relu(self.model.norm2(self.model.fc2(encoded_embedding)))
        quantized, commitment_loss = self.model.vq(norm2_output)
        return quantized, commitment_loss

    @torch.no_grad()
    def generate(
        self,
        input_embeddings_batch,
        input_masks_batch,
        input_masks_invert,
        target_ids_batch_converted,
        max_length,
        num_beams,
        repetition_penalty,
    ):
        encoded_embedding, _ = self.addin_forward(
            input_embeddings_batch, input_masks_invert
        )
        output = self.model.llm.generate(
            # encoder_outputs
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, : encoded_embedding.shape[1]],
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            # labels=target_ids_batch_converted,
            # return_dict_in_generate=True,
        )
        return output

    def forward(
        self,
        input_embeddings_batch,
        input_masks_batch,
        input_masks_invert,
        target_ids_batch_converted,
    ):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        addin_output, commitment_loss = self.addin_forward(
            input_embeddings_batch, input_masks_invert
        )
        output = self.model.llm(
            inputs_embeds=addin_output,
            attention_mask=input_masks_batch,
            return_dict=True,
            labels=target_ids_batch_converted,
        )

        return output, commitment_loss

# class Encoder(nn.Module):
#     def __init__(self, embedding_dim=840, hidden_dim=1024):
#         super().__init__()
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         outputs, (hidden, cell) = self.lstm(x)
#         return outputs, (hidden, cell)

# class Decoder(nn.Module):
#     def __init__(self, embedding_dim=840, hidden_dim=1024, target_vocab_size=2048):
#         super().__init__()
#         self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim+hidden_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, target_vocab_size)
    
#     def forward(self, x, hidden, cell):
#         # x = x.unsqueeze(1)
#         x = self.embedding(x)
#         context = hidden[-1].repeat(x.shape[1], 1, 1).permute(1, 0, 2)
#         # print(context.shape)
#         x_and_context = torch.cat((x, context), 2)
#         outputs, (hidden, cell) = self.lstm(x_and_context, (hidden, cell))
#         x = self.fc(outputs)
#         # print(x.shape)
#         return x, (hidden, cell)

class EEGToWord(nn.Module):
    def __init__(self, input_dim=840, hidden_dim=2048, vocab_size=1000, max_seq_length=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.max_len = max_seq_length
        
    def forward(self, x):
        # output = self.net(x)
        # return output.reshape(output.shape[0],self.max_len,int(output.shape[1]/self.max_len))
        return self.net(x)

class Quantize(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=1024, decay=0.99, eps=1e-5):
        super(Quantize, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        codebook = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", codebook.clone())

    def forward(self, input):
        # Flatten input tensor
        # input [32,56,1024]
        batch_size, sequence_length, input_dim = input.size()
        input_flat = input.view(-1, input_dim)  # [32*56,1024]

        # Compute distances between input and embeddings
        distances = torch.cdist(input_flat, self.codebook, p=2)  # [1792,512]

        # Find closest embedding for each input vector
        _, indexs = torch.min(distances, dim=1)

        # Quantize the input
        quantized = self.embed_code(indexs).view(batch_size, sequence_length, input_dim)

        # update codebook
        embed_onehot = F.one_hot(indexs, self.num_embeddings).type(input_flat.dtype)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = input_flat.transpose(0, 1) @ embed_onehot
        self.cluster_size.data.mul_(self.decay).add_(
            embed_onehot_sum, alpha=1 - self.decay
        )
        self.embed_avg.data.mul_(self.decay).add_(
            embed_sum.transpose(0, 1), alpha=1 - self.decay
        )
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
        )
        embed_normalized = self.embed_avg.transpose(0, 1) / cluster_size.unsqueeze(0)
        self.codebook.data.copy_(embed_normalized.transpose(0, 1))

        # Compute the commitment loss
        commitment_loss = ((quantized.detach() - input).pow(2)).mean()

        # Straight-through estimator
        quantized = input + (quantized - input).detach()

        return quantized, commitment_loss

    def embed_code(self, embed_index):
        return F.embedding(embed_index, self.codebook)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)