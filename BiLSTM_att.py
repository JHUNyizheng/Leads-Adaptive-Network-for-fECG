# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile  # For calculating FLOPs
import matplotlib.pyplot as plt
import copy


class CombinedLSTM(nn.Module):
    # Model definition remains unchanged
    def __init__(self, input_size, hidden_size=128, num_layers=2, std=0.1, l2_reg=0.001, window_size=1024):
        super(CombinedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.l2_reg = l2_reg
        self.window_size = window_size

        # Single-channel BiLSTM layers
        self.lstm_list = nn.ModuleList([
            nn.LSTM(1, hidden_size, num_layers, batch_first=True, bidirectional=True)
            for _ in range(input_size)
        ])

        # Manually apply Xavier initialization with adjusted gain
        gain = 1  # Set a smaller gain value
        for lstm in self.lstm_list:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=gain)
                if 'bias' in name:
                    nn.init.zeros_(param)

        # Regression layer for each channel
        self.channel_regressors = nn.ModuleList([
                nn.Linear(2 * hidden_size, 1)
            for _ in range(input_size)
        ])

        # Channel self-attention layer (sequence layer sliding window)
        self.attention2 = nn.Sequential(
            nn.Linear(window_size, window_size//2),
            nn.ReLU(),
            nn.Linear(window_size//2, 1)
        )
        for m in self.attention2.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Channel self-attention layer (hidden layer)
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Fusion BiLSTM layer
        self.fusion_lstm = nn.LSTM(
            2 * hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True
        )

        # Initialize fusion LSTM
        for name, param in self.fusion_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=gain)
            if 'bias' in name:
                nn.init.zeros_(param)

        # Regression prediction layer and MLP decoding layer
        self.regressor = nn.Linear(2*hidden_size, 1)
        self.MLP_decoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, x):
        enhance = False
        frequency = False
        time_domain = True
        max = True
        threshold = 0.05

        if enhance:
            noise = torch.randn_like(x) * 0.1
            x = x + noise

        all_outputs = []
        all_channel_preds = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Process each channel individually with BiLSTM
        for i in range(self.input_size):
            channel_input = x[:, i:i + 1, :].transpose(1, 2)  # [batch, seq, 1]
            h0 = torch.zeros(2 * self.num_layers, channel_input.size(0), self.hidden_size, device=device)
            c0 = torch.zeros(2 * self.num_layers, channel_input.size(0), self.hidden_size, device=device)
            output, _ = self.lstm_list[i](channel_input, (h0, c0))
            all_outputs.append(output)

            # Prediction for each channel
            channel_pred = self.channel_regressors[i](output).squeeze(-1)
            all_channel_preds.append(channel_pred)

        channel_outputs = torch.stack(all_outputs, dim=2)  # [batch, seq, ch, hidden]

        # Calculate attention weights using sliding window; window length equals signal length
        _, seq_length, _, _ = channel_outputs.shape
        num_windows = seq_length // self.window_size
        all_alphas = []
        for i in range(num_windows):
            start_idx = i * self.window_size
            end_idx = (i + 1) * self.window_size
            window_outputs = channel_outputs[:, start_idx:end_idx, :, :]
            sel1 = window_outputs.permute(0, 3, 2, 1)
            attention_scores = self.attention2(sel1)
            alpha1 = attention_scores.expand_as(sel1)
            all_alphas.append(alpha1)

        alpha2 = torch.cat(all_alphas, dim=3).permute(0, 3, 2, 1)
        alpha3 = self.attention(alpha2)
        alpha = torch.softmax(alpha3, dim=2)
        select_channels = (channel_outputs * alpha).sum(dim=2)
        fused_outputs = select_channels
        # MLP decoding
        pred = self.MLP_decoder(fused_outputs).squeeze(-1)

        # Calculate L2 regularization term
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param) **2
        l2_loss = self.l2_reg * l2_loss

        return pred, alpha, l2_loss, all_channel_preds

##Test if the model works properly
# Initialize the model
input_size = 4
hidden_size = 128
num_layers = 2
std = 1e-2
l2_reg = 0.001
model = CombinedLSTM(input_size, hidden_size, num_layers, std, l2_reg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 1. Calculate total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # With thousands separator

# 2. Use torchinfo to print parameter details of each layer (compatible with LSTM tuple output)
print("\nParameter details of each layer:")
# torchinfo's summary function has more flexible parameters and automatically handles complex outputs
summary(
    model,
    input_size=(1, input_size, 1024),  # (batch_size, input_size, sequence_length)
    device=device.type,
    col_names=["input_size", "output_size", "num_params", "mult_adds"],  # Columns to display
    col_width=20,
    depth=4  # Display hierarchy depth
)

# 3. Use thop to calculate FLOPs and parameter count
batch_size = 32
sequence_length = 1024
input_data = torch.randn(batch_size, input_size, sequence_length).to(device)
flops, params = profile(model, inputs=(input_data,))
print(f"\nFLOPs: {flops / 1e9:.2f} G")  # Convert to GFlops
print(f"Parameter count (thop): {params / 1e6:.2f} M")  # Convert to million parameters

# Model test
model.eval()
with torch.no_grad():
    output, alpha, l2_loss, channel_preds = model(input_data)

# Calculate loss
criterion = nn.MSELoss()
total_channel_loss = sum(criterion(pred, torch.randn_like(pred)) for pred in channel_preds)

print("\nOutput shape:", output.shape)
print("Total loss of each channel:", total_channel_loss.item())
print("L2 regularization loss:", l2_loss.item())