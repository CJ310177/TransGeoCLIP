# Note: This code is a draft version. 
# Full documentation and final optimizations will be updated upon official acceptance.


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from .rff.layers import GaussianEncoding  # Custom RFF layer assumed available
# from pyproj import Proj, Transformer      # Geospatial projection library

# Define path to local pre-trained CLIP model
local_model_dir = 'your path'

class LocationTransformer(nn.Module):
    """
    A simple Transformer encoder for processing location data.
    Projects input coordinates to a hidden dimension, applies self-attention, 
    and outputs a refined feature vector.
    """
    def __init__(self, input_dim=2, hidden_dim=256, num_heads=8, num_layers=3):
        super(LocationTransformer, self).__init__()
        # Linear projection to embed input coordinates into hidden dimension
        self.pos_encoder = nn.Linear(input_dim, hidden_dim)
        
        # Define Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        # Stack encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Final output projection
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # Project input to hidden dimension


class LocationAttention(nn.Module):
    """
    Custom attention mechanism
    """
    def __init__(self, dim=512):
        super(LocationAttention, self).__init__()
        # Linear layers for Query, Key, Value
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # Scaling factor for attention scores
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(dim, dtype=torch.float32)))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, distances):
        # x: [batch, dim], distances: [batch, batch]
        # Compute Q, K, V
        q = self.query(x).unsqueeze(1)
        k = self.key(x).unsqueeze(0)    
        v = self.value(x).unsqueeze(0)  
        output = # waiting
        # Residual connection
        return output + x

class LocationEncoderCapsule(nn.Module):
    """
    A single encoding capsule that uses Random Fourier Features (RFF) 
    followed by a Transformer to encode spatial coordinates.
    """
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        # RFF encoding to map 2D coordinates to a higher dimensional space
        self.rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        # Transformer to process the encoded features
        self.transformer = LocationTransformer(
            input_dim=512, hidden_dim=1024, num_heads=8, num_layers=3
        )
        # Head to project to final feature dimension
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        # Apply RFF encoding
        x = self.rff_encoding(x)  
        # Add sequence dimension for Transformer
        x = x.unsqueeze(1)  
        # Encode with Transformer
        x = self.transformer(x)  
        # Project to output size
        return self.head(x)  

class CustomLocationEncoder(nn.Module):
    """
    Aggregates multiple LocationEncoderCapsules with different scales (sigmas)
    and applies distance-aware attention to fuse the features.
    Includes coordinate projection from Lat/Lon to Web Mercator.
    """
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(CustomLocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        
        # Create multiple encoder capsules with different frequency scales
        for i, s in enumerate(self.sigma):
            self.add_module(f'LocEnc{i}', LocationEncoderCapsule(sigma=s))
        
        # Attention module to fuse features
        self.location_attention = LocationAttention(dim=512)
        
        # Coordinate transformer: EPSG:4326 (Lat/Lon) to EPSG:3857 (Web Mercator)
        self.transformer = Transformer.from_proj(
            Proj('epsg:4326'), Proj('epsg:3857'), always_xy=True
        )

    def forward(self, input):
        # Extract latitude and longitude
        lat, lon = input[:, 0].float(), input[:, 1].float()
        
        # Project coordinates to Web Mercator (requires CPU numpy operation)
        projected = self.transformer.transform(lon.detach().cpu().numpy(), 
                                               lat.detach().cpu().numpy())
        
        # Convert back to tensor and normalize to [-1, 1] range approx
        location = torch.tensor(list(zip(*projected)))[:, [1, 0]].to(input.device)
        location = location / 20037508.3427892 
        
        # Calculate pairwise Euclidean distances between all locations in batch
        distances = torch.cdist(location, location, p=2) 
        
        # Sum features from all capsules (multi-scale encoding)
        location_features = torch.zeros(location.shape[0], 512).to(input.device)
        for i in range(self.n):
            location_features += self._modules[f'LocEnc{i}'](location)
        
        # Apply distance-aware attention
        return self.location_attention(location_features, distances)

class TransGeoCLIP(torch.nn.Module):
    """
    Main Model: TransGeoCLIP
    Integrates CLIP (Vision & Text) with a custom Geographic Location Encoder.
    Trains using multi-phase contrastive losses (Image-Text, Image-Location, Text-Location)
    and a triplet loss to align all three modalities.
    """
    def __init__(self, device):
        super(TransGeoCLIP, self).__init__()
        self.device = device

        # Load pre-trained CLIP model components
        clip_model = CLIPModel.from_pretrained(local_model_dir)
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.vision_processor = CLIPImageProcessor.from_pretrained(local_model_dir)
        self.text_processor = CLIPTokenizer.from_pretrained(local_model_dir)
        self.vision_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection

        # Trainable projection heads for fine-tuning modality alignment
        self.vision_proj1 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.vision_proj2 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.text_proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.text_proj_loc = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 512))
        self.location_proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 768))

        # Learnable temperature parameters for contrastive loss scaling

        # Initialize custom location encoder
        self.location_encoder = CustomLocationEncoder()
        
        # Freeze base CLIP models to prevent weight updates during training
        for model in [self.vision_model, self.vision_projection, 
                      self.text_model, self.text_projection]:
            model.requires_grad_(False)

    def _normalize(self, *tensors):
        # L2 normalize tensors along the last dimension
        return [t / t.norm(p=2, dim=-1, keepdim=True) for t in tensors]

    def _compute_similarity(self, emb1, emb2, scale_name):
        # Compute cosine similarity matrix scaled by learnable temperature
        logit_scale = self.logit_scales[scale_name].exp()
        return torch.matmul(emb1, emb2.t()) * logit_scale

    def forward(self, images, texts, longitude, latitude, return_loss=True):
        # Extract base embeddings from frozen CLIP models
        vision_output = self.vision_model(images)[1]
        text_output = self.text_model(**texts)[1]
        image_embeds = self.vision_projection(vision_output) 
        text_embeds = self.text_projection(text_output)      

        # Generate location embeddings using custom encoder
        locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(locations) 

        # Phase 1: Image-Text Contrastive Learning
        img1, txt1 = self._normalize(self.vision_proj1(image_embeds), 
                                      self.text_proj(text_embeds))
        sim_it = self._compute_similarity(txt1, img1, 't1')
        
        # Phase 2: Image-Location Contrastive Learning
        img2, loc2 = self._normalize(self.vision_proj2(image_embeds),
                                      self.location_proj(location_embeds))
        sim_il = self._compute_similarity(loc2, img2, 't2')
        
        # Phase 3: Text-Location Contrastive Learning
        txt3, loc3 = self._normalize(self.text_proj_loc(text_embeds), location_embeds)
        sim_tl = self._compute_similarity(txt3, loc3, 't3')

        losses = {}
        if return_loss:
            # Calculate symmetric contrastive losses for each pair
            losses['p1'] = self.clip_loss(sim_it)
            losses['p2'] = self.clip_loss(sim_il)
            losses['p3'] = self.clip_loss(sim_tl)
            
            # Phase 4: Triplet Loss (Text Anchor, Location Positive, Image Negative)
            img_trip, txt_trip, loc_trip = self._normalize(
                self.vision_proj1(image_embeds)[:, :512],
                self.text_proj_loc(text_embeds),
                location_embeds
            )
            losses['p4'] = 0.5 * self.triplet_loss(txt_trip, loc_trip, img_trip, margin=0.5)

        # Return similarities, total loss, and raw embeddings
        return {
            'sim_image_text': sim_it,
            'sim_text_image': sim_it.t(),
            'sim_image_location': sim_il,
            'sim_location_image': sim_il.t(),
            'sim_text_location': sim_tl,
            'sim_location_text': sim_tl.t(),
            'loss': sum(losses.values()) if losses else None,
            'embeddings': {
                'image': image_embeds, 'text': text_embeds, 'location': location_embeds
            }
        }

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy loss for contrastive learning (diagonal targets)
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        # Symmetric CLIP loss: average of row-wise and column-wise cross-entropy
        return (self.contrastive_loss(similarity) + 
                self.contrastive_loss(similarity.t())) / 2.0

    def triplet_loss(self, anchor, positive, negative, margin=0.5):
        # Compute triplet loss with hardest negative mining
        batch_size = anchor.size(0)
        # Similarity between anchor and positive
        pos_sim = torch.sum(anchor * positive, dim=-1)
        
        # Similarity between anchor and all negatives
        neg_sim = torch.matmul(anchor, negative.t())
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=anchor.device).bool()
        neg_sim.masked_fill_(mask, -float('inf'))
        # Select hardest negative (max similarity among negatives)
        hardest_neg = neg_sim.max(dim=1)[0]
        
        # Hinge loss
        losses = torch.clamp(margin + hardest_neg - pos_sim, min=0.0)
        return losses.mean()
