# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 20:10:45 2025

@author: Gavin
"""

import torch
import torch.nn as nn


class TextMLP(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_classes=2, dropout=0.5):
        super(TextMLP, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class TextCNN(nn.Module):  
    def __init__(self, input_dim, num_filters=50, filter_sizes=[3, 4], num_classes=2, dropout=0.5):  
        """  
        Optimized TextCNN without an embedding layer. This assumes the input is already a fixed-size vector.  
  
        Args:  
            input_dim (int): Size of the input feature vector (e.g., vocabulary size for BoW or TF-IDF).  
            num_filters (int): Reduced number of filters for each convolutional layer.  
            filter_sizes (list): Reduced list of filter sizes (kernel sizes) for the convolutional layers.  
            num_classes (int): Number of output classes (e.g., positive/negative sentiment).  
            dropout (float): Dropout probability for regularization.  
        """  
        super(TextCNN, self).__init__()  
  
        # Convolutional layers with fewer filters and filter sizes  
        self.convs = nn.ModuleList([  
            nn.Conv1d(1, num_filters, kernel_size=fs, padding=fs // 2)  # Padding ensures consistent output sizes  
            for fs in filter_sizes  
        ])  
  
        # Dropout and final classifier  
        self.dropout = nn.Dropout(dropout)  
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_classes)  
  
    def forward(self, x):  
        """  
        Forward pass through the model.  
  
        Args:  
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).  
  
        Returns:  
            torch.Tensor: Output logits of shape (batch_size, num_classes).  
        """  
        # Add a channel dimension for CNN (as required by Conv1d)  
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)  
  
        # Apply convolutions and pooling  
        conv_outputs = [  
            torch.max_pool1d(torch.relu(conv(x)), x.size(2)).squeeze(2)  # Faster pooling  
            for conv in self.convs  
        ]  
  
        # Concatenate all pooled outputs  
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)  
  
        # Apply dropout  
        x = self.dropout(x)  
  
        # Final classification layer  
        return self.classifier(x)  