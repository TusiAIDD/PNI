import math
import torch

from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn

# from mamba_ssm import Mamba2


class PNI_FCN(nn.Module):
    def __init__(self, params):
        super(PNI_FCN, self).__init__()

        self.modelName = params['modelName']

        self.peptide_fc = nn.Sequential(
            nn.Linear(params['peptide_length'] * 22 if params['mode'] == 'OH' else 20, 512),
            nn.ReLU(),
            nn.Linear(512, params['hidden_dim']),
            nn.ReLU(),
        )

        self.ligand_fc = nn.Sequential(
            nn.Linear(params['ligand_length'] * 5, 512),
            nn.ReLU(),
            nn.Linear(512, params['hidden_dim']),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(params['hidden_dim'] * 2, params['hidden_dim']),
            nn.ReLU(),
            nn.Linear(params['hidden_dim'], 2),
        )

    def forward(self, peptide, ligand):
        peptide_representation = self.peptide_fc(peptide.view(peptide.size(0), -1))
        ligand_representation = self.ligand_fc(ligand.view(ligand.size(0), -1))
        combined = torch.cat((peptide_representation, ligand_representation), dim=1)
        x = self.fc(combined)
        return x, None


class BindingSiteAttention(nn.Module):
    def __init__(self, d_model, seq_length):
        super(BindingSiteAttention, self).__init__()
        self.seq_length = seq_length
        self.query = nn.Parameter(torch.randn(d_model))
        self.key = nn.Linear(d_model, d_model)

    def forward(self, encoder_outputs, mask):
        # 计算注意力分数
        keys = self.key(encoder_outputs)  # [batch_size, seq_length, d_model]
        attn_scores = torch.einsum('bnd,d->bn', keys, self.query)  # [batch_size, seq_length]
        attn_scores = attn_scores.masked_fill(mask == False, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_length]

        # 应用注意力池化
        pooled_output = torch.bmm(attn_weights, encoder_outputs).squeeze(1)  # [batch_size, d_model]
        return pooled_output, attn_weights.squeeze(1)

# 同时接受peptide与lignand的注意力机制
# class BindingSiteAttention(nn.Module):
#     def __init__(self, d_model, pep_len, ligand_len):
#         super(BindingSiteAttention, self).__init__()
#         # self.seq_length = seq_length
#         self.query = nn.Linear(d_model, d_model)  # 使用peptide的输入维度
#         self.key = nn.Linear(d_model, d_model)  # 使用合并后的输入维度
#         self.value = nn.Linear(d_model, d_model)  # 使用peptide的输入维度
#
#         self.key_trans = nn.Linear(pep_len + ligand_len, pep_len)
#
#     def forward(self, peptide_outputs, ligand_outputs, mask):
#         # 将peptide和ligand的输出合并,用于计算key
#         combined_outputs = torch.cat([peptide_outputs, ligand_outputs], dim=1)  # [batch_size, seq_length, 2 * d_model]
#
#         # 计算qkv
#         queries = self.query(peptide_outputs)  # [batch_size, seq_length, d_model]
#         keys = self.key(combined_outputs).transpose(1, 2)  # [batch_size, seq_length, d_model]
#         keys = self.key_trans(keys).transpose(1, 2)
#         values = self.value(peptide_outputs)  # [batch_size, seq_length, d_model]
#
#         # 计算注意力分数
#         attn_scores = torch.einsum('bnd,bnd->bn', queries, keys)  # [batch_size, seq_length]
#         attn_scores = attn_scores.masked_fill(mask == False, float('-inf'))
#         attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_length]
#
#         # 应用注意力池化
#         pooled_output = torch.bmm(attn_weights, values).squeeze(1)  # [batch_size, d_model]
#
#         return pooled_output, attn_weights.squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach().to(x.device)


class PNI_transformer(nn.Module):
    def __init__(self, params):
        super(PNI_transformer, self).__init__()

        self.modelName = params['modelName']

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params['d_model'],
            nhead=params['nhead'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'],
            batch_first=True)

        self.peptide_embedding = nn.Conv1d(in_channels=22 if params['mode'] == 'OH' else 20, out_channels=params['d_model'], kernel_size=1)
        self.peptide_positional_encoding = PositionalEncoding(params['d_model'])
        self.peptide_layer = nn.TransformerEncoder(encoder_layer, num_layers=params['num_encoder_layers'])
        self.peptide_binding_site_attention = BindingSiteAttention(params['d_model'], seq_length=params['peptide_length'])
        # self.peptide_binding_site_attention = BindingSiteAttention(params['d_model'], params['peptide_length'], params['ligand_length'])

        self.ligand_embedding = nn.Conv1d(in_channels=5, out_channels=params['d_model'], kernel_size=1)
        self.ligand_positional_encoding = PositionalEncoding(params['d_model'])
        self.ligand_layer = nn.TransformerEncoder(encoder_layer, num_layers=params['num_encoder_layers'])
        self.ligand_attention = BindingSiteAttention(params['d_model'], seq_length=params['ligand_length'])
        # self.ligand_attention = BindingSiteAttention(params['d_model'], params['peptide_length'], params['ligand_length'])

        self.fc = nn.Sequential(
            nn.Linear(params['d_model'] * 2, params['d_model']),
            nn.ReLU(),
            nn.Linear(params['d_model'], 2),
        )

    def forward(self, peptide, ligand):
        peptide_mask = (peptide != 0).any(dim=-1)
        peptide = self.peptide_embedding(peptide.float().permute(0, 2, 1)).permute(0, 2, 1)
        peptide = self.peptide_positional_encoding(peptide)
        peptide = self.peptide_layer(peptide)

        ligand_mask = (ligand != 0).any(dim=-1)
        ligand = self.ligand_embedding(ligand.float().permute(0, 2, 1)).permute(0, 2, 1)
        ligand = self.ligand_positional_encoding(ligand)
        ligand = self.ligand_layer(ligand)

        peptide_representation, site = self.peptide_binding_site_attention(peptide, peptide_mask)
        ligand_representation, _ = self.ligand_attention(ligand, ligand_mask)

        combined = torch.cat((peptide_representation, ligand_representation), dim=1)
        x = self.fc(combined)
        return x, site


class MambaNetwork(nn.Module):
    def __init__(self, num_blocks, d_model, d_state, d_conv, expand):
        super(MambaNetwork, self).__init__()
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PNI_mamba(nn.Module):
    """
    纯MMAMBA块堆的网络
    """
    def __init__(self, params):
        super(PNI_mamba, self).__init__()

        self.modelName = params['modelName']

        self.peptide_embedding = nn.Conv1d(in_channels=22 if params['mode'] == 'OH' else 20, out_channels=params['d_model'], kernel_size=1)
        self.peptide_positional_encoding = PositionalEncoding(params['d_model'])
        self.peptide_layer = MambaNetwork(
                        num_blocks=1,
                        d_model=params['d_model'], # Model dimension d_model
                        d_state=params['d_state'],  # SSM state expansion factor4
                        d_conv=params['d_conv'],    # Local convolution width2
                        expand=params['expand'],    # Block expansion factor8
                        )
        self.peptide_binding_site_attention = BindingSiteAttention(params['d_model'], seq_length=params['peptide_length'])
        # self.peptide_binding_site_attention = BindingSiteAttention(params['d_model'], params['peptide_length'], params['ligand_length'])

        self.ligand_embedding = nn.Conv1d(in_channels=5, out_channels=params['d_model'], kernel_size=1)
        self.ligand_positional_encoding = PositionalEncoding(params['d_model'])
        self.ligand_layer = MambaNetwork(
                        num_blocks=1,
                        d_model=params['d_model'], # Model dimension d_model
                        d_state=params['d_state'],  # SSM state expansion factor
                        d_conv=params['d_conv'],    # Local convolution width
                        expand=params['expand'],    # Block expansion factor
                        )
        self.ligand_attention = BindingSiteAttention(params['d_model'], seq_length=params['ligand_length'])
        # self.ligand_attention = BindingSiteAttention(params['d_model'], params['ligand_length'], params['peptide_length'])

        self.fc = nn.Sequential(
            nn.Linear(params['d_model'] * 2, params['d_model']),
            nn.ReLU(),
            nn.Linear(params['d_model'], 2),
        )

        self.dp = params['dp']
        self.dropout = nn.Dropout(p=self.dp)

    def forward(self, peptide, ligand):
        peptide_mask = (peptide != 0).any(dim=-1)
        peptide = self.peptide_embedding(peptide.float().permute(0, 2, 1)).permute(0, 2, 1)

        if self.dp > 0.0:
            peptide = self.dropout(peptide)

        # peptide = self.peptide_positional_encoding(peptide)
        peptide = self.peptide_layer(peptide)

        if self.dp > 0.0:
            peptide = self.dropout(peptide)

        ligand_mask = (ligand != 0).any(dim=-1)
        ligand = self.ligand_embedding(ligand.float().permute(0, 2, 1)).permute(0, 2, 1)

        if self.dp > 0.0:
            ligand = self.dropout(ligand)

        # ligand = self.ligand_positional_encoding(ligand)
        ligand = self.ligand_layer(ligand)

        if self.dp > 0.0:
            ligand = self.dropout(ligand)

        # peptide_representation, site = self.peptide_binding_site_attention(peptide, ligand, peptide_mask)
        peptide_representation, site = self.peptide_binding_site_attention(peptide, peptide_mask)
        if self.dp > 0.0:
            peptide_representation = self.dropout(peptide_representation)

        # ligand_representation, _ = self.ligand_attention(ligand, peptide, ligand_mask)
        ligand_representation, _ = self.ligand_attention(ligand, ligand_mask)
        if self.dp > 0.0:
            ligand_representation = self.dropout(ligand_representation)

        combined = torch.cat((peptide_representation, ligand_representation), dim=1)
        x = self.fc(combined)
        return x, site

    def init_weights(self, m):
        if hasattr(m, 'weight'):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MambaNetwork2(nn.Module):
    def __init__(self, num_blocks, d_model, d_state, d_conv, expand):
        super(MambaNetwork2, self).__init__()
        self.layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, A_init_range=(1, 1.1))       # , chunk_size=1
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PNI_mamba2(nn.Module):
    """
    纯MMAMBA块堆的网络
    """
    def __init__(self, params):
        super(PNI_mamba2, self).__init__()

        self.modelName = params['modelName']

        self.peptide_embedding = nn.Conv1d(in_channels=22 if params['mode'] == 'OH' else 20, out_channels=params['d_model'], kernel_size=1)
        self.peptide_positional_encoding = PositionalEncoding(params['d_model'])
        self.peptide_layer = MambaNetwork2(
                        num_blocks=1,
                        d_model=params['d_model'], # Model dimension d_model
                        d_state=params['d_state'],  # SSM state expansion factor4
                        d_conv=params['d_conv'],    # Local convolution width2
                        expand=params['expand'],    # Block expansion factor8
                        )
        self.peptide_binding_site_attention = BindingSiteAttention(params['d_model'], seq_length=params['peptide_length'])
        # self.peptide_binding_site_attention = BindingSiteAttention(params['d_model'], params['peptide_length'], params['ligand_length'])

        self.ligand_embedding = nn.Conv1d(in_channels=5, out_channels=params['d_model'], kernel_size=1)
        self.ligand_positional_encoding = PositionalEncoding(params['d_model'])
        self.ligand_layer = MambaNetwork2(
                        num_blocks=1,
                        d_model=params['d_model'], # Model dimension d_model
                        d_state=params['d_state'],  # SSM state expansion factor
                        d_conv=params['d_conv'],    # Local convolution width
                        expand=params['expand'],    # Block expansion factor
                        )
        self.ligand_attention = BindingSiteAttention(params['d_model'], seq_length=params['ligand_length'])
        # self.ligand_attention = BindingSiteAttention(params['d_model'], params['ligand_length'], params['peptide_length'])

        self.fc = nn.Sequential(
            nn.Linear(params['d_model'] * 2, params['d_model']),
            nn.ReLU(),
            nn.Linear(params['d_model'], 2),
        )

        self.dp = params['dp']
        self.dropout = nn.Dropout(p=self.dp)

    def forward(self, peptide, ligand):
        peptide_mask = (peptide != 0).any(dim=-1)
        peptide = self.peptide_embedding(peptide.float().permute(0, 2, 1)).permute(0, 2, 1)

        if self.dp > 0.0:
            peptide = self.dropout(peptide)

        # peptide = self.peptide_positional_encoding(peptide)
        peptide = self.peptide_layer(peptide)

        if self.dp > 0.0:
            peptide = self.dropout(peptide)

        ligand_mask = (ligand != 0).any(dim=-1)
        ligand = self.ligand_embedding(ligand.float().permute(0, 2, 1)).permute(0, 2, 1)

        if self.dp > 0.0:
            ligand = self.dropout(ligand)

        # ligand = self.ligand_positional_encoding(ligand)
        ligand = self.ligand_layer(ligand)

        if self.dp > 0.0:
            ligand = self.dropout(ligand)

        # peptide_representation, site = self.peptide_binding_site_attention(peptide, ligand, peptide_mask)
        peptide_representation, site = self.peptide_binding_site_attention(peptide, peptide_mask)
        if self.dp > 0.0:
            peptide_representation = self.dropout(peptide_representation)

        # ligand_representation, _ = self.ligand_attention(ligand, peptide, ligand_mask)
        ligand_representation, _ = self.ligand_attention(ligand, ligand_mask)
        if self.dp > 0.0:
            ligand_representation = self.dropout(ligand_representation)

        combined = torch.cat((peptide_representation, ligand_representation), dim=1)
        x = self.fc(combined)
        return x, site

    def init_weights(self, m):
        if hasattr(m, 'weight'):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

