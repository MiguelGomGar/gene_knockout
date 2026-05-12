import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#%% MLP Regressor architectures
class OriginalMLPRegressor(nn.Module):
    def __init__(self, d=10716, hidden_dim_dis=256, hidden_dim_int=64, drop_rate=0.2):
        super().__init__()
        
        # Encoder for the diseased inputs
        self.encoder_dis = nn.Sequential(
            nn.Linear(d, hidden_dim_dis),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Encoder for the the intervention inputs
        self.encoder_int = nn.Sequential(
            nn.Linear(d, hidden_dim_int),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Combines the outputs of both encoders and predicts the outcome
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim_dis + hidden_dim_int, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, d) 
        )

    def forward(self, diseased, intervention):
        h_dis = self.encoder_dis(diseased)
        h_int = self.encoder_int(intervention)
        h_cat = torch.cat([h_dis, h_int], dim=1)
        delta = self.prediction_head(h_cat)
        
        return delta

class NormalizerMLPRegressor(nn.Module):
    def __init__(self, d=10716, hidden_dim_dis=256, hidden_dim_int=64, drop_rate=0.2):
        super().__init__()
        
        # Encoder for the diseased inputs
        self.encoder_dis = nn.Sequential(
            nn.Linear(d, hidden_dim_dis),
            nn.BatchNorm1d(hidden_dim_dis),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Encoder for the intervention inputs
        self.encoder_int = nn.Sequential(
            nn.Linear(d, hidden_dim_int),
            nn.BatchNorm1d(hidden_dim_int),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Combines the outputs of both encoders and predicts the outcome
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim_dis + hidden_dim_int, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, d) 
        )

    def forward(self, diseased, intervention):
        h_dis = self.encoder_dis(diseased)
        h_int = self.encoder_int(intervention)
        h_cat = torch.cat([h_dis, h_int], dim=1)
        delta = self.prediction_head(h_cat)
        
        return delta

class SkipConnMLPRegressor(nn.Module):
    def __init__(self, d=10716, hidden_dim_dis=256, hidden_dim_int=64, drop_rate=0.2):
        super().__init__()
        
        # Encoder for the diseased inputs
        self.encoder_dis = nn.Sequential(
            nn.Linear(d, hidden_dim_dis),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Encoder for the intervention inputs
        self.encoder_int = nn.Sequential(
            nn.Linear(d, hidden_dim_int),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Combines the outputs of both encoders and predicts the outcome
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim_dis + hidden_dim_int, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, d) 
        )

    def forward(self, diseased, intervention):
        h_dis = self.encoder_dis(diseased)
        h_int = self.encoder_int(intervention)
        h_cat = torch.cat([h_dis, h_int], dim=1)
        delta = self.prediction_head(h_cat)
        
        # The model predicts the change from the diseased state and adds it to the original diseased state
        return diseased + delta

class UltimateMLPRegressor(nn.Module):
    def __init__(self, d=10716, hidden_dim_dis=256, hidden_dim_int=64, drop_rate=0.2):
        super().__init__()
        
        # Encoder for the diseased inputs
        self.encoder_dis = nn.Sequential(
            nn.Linear(d, hidden_dim_dis),
            nn.BatchNorm1d(hidden_dim_dis),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Encoder for the intervention inputs
        self.encoder_int = nn.Sequential(
            nn.Linear(d, hidden_dim_int),
            nn.BatchNorm1d(hidden_dim_int),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # Combines the outputs of both encoders and predicts the outcome
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim_dis + hidden_dim_int, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, d) 
        )

    def forward(self, diseased, intervention):
        h_dis = self.encoder_dis(diseased)
        h_int = self.encoder_int(intervention)
        h_cat = torch.cat([h_dis, h_int], dim=1)
        delta = self.prediction_head(h_cat)
        
        # The model predicts the change from the diseased state and adds it to the original diseased state
        return diseased + delta

#%% Transformer Regressor architecture