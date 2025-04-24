import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import time

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configure paths for macOS
DATA_DIR = './Dataset/'
OUTPUT_DIR = './output/'

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions for data processing
def load_data():
    """Load and return all necessary Instacart datasets"""
    print("Loading datasets...")
    orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    order_products_prior = pd.read_csv(os.path.join(DATA_DIR, 'order_products__prior.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    aisles = pd.read_csv(os.path.join(DATA_DIR, 'aisles.csv'))
    departments = pd.read_csv(os.path.join(DATA_DIR, 'departments.csv'))
    
    # Display basic info
    print(f"Orders: {orders.shape}")
    print(f"Order products (prior): {order_products_prior.shape}")
    print(f"Products: {products.shape}")
    
    return orders, order_products_prior, products, aisles, departments

def create_cyclic_features(df):
    """Create cyclic encodings for time features"""
    # Hour of day (0-23) -> sin/cos encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['order_hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['order_hour_of_day'] / 24)
    
    # Day of week (0-6) -> sin/cos encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['order_dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['order_dow'] / 7)
    
    return df

def preprocess_data(orders, order_products_prior, products, max_session_length=20):
    """
    Preprocess the Instacart data to create session sequences with temporal features
    """
    print("Preprocessing data...")
    
    # Add cyclic features to orders
    orders = create_cyclic_features(orders)
    
    # Merge order products with orders to get temporal features
    order_data = order_products_prior.merge(
        orders[['order_id', 'user_id', 'order_number', 'order_hour_of_day', 'order_dow', 'days_since_prior_order',
               'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']], 
        on='order_id'
    )
    
    # Sort by user and order
    order_data = order_data.sort_values(by=['user_id', 'order_number', 'add_to_cart_order'])
    
    # Create sequences per order (session)
    sessions = []
    temporal_features = []
    
    # Process each user's orders
    for user_id, user_data in tqdm(order_data.groupby('user_id'), desc="Creating sessions"):
        for order_id, order_items in user_data.groupby('order_id'):
            # Get product sequence
            product_seq = order_items['product_id'].tolist()
            
            # Skip if sequence is too short (need at least 2 items for next-item prediction)
            if len(product_seq) < 2:
                continue
                
            # Truncate if sequence is too long
            if len(product_seq) > max_session_length:
                product_seq = product_seq[:max_session_length]
            
            # Get temporal features (use first item's temporal data for the whole order)
            temp_features = order_items.iloc[0][
                ['order_hour_of_day', 'order_dow', 'days_since_prior_order', 
                 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
            ].to_dict()
            
            # Handle NaN in days_since_prior_order (first order)
            if np.isnan(temp_features['days_since_prior_order']):
                temp_features['days_since_prior_order'] = 0
                
            sessions.append(product_seq)
            temporal_features.append(temp_features)
    
    print(f"Created {len(sessions)} sessions")
    
    # Create product mapping (product_id -> index)
    unique_products = order_data['product_id'].unique()
    product_to_idx = {pid: idx+1 for idx, pid in enumerate(unique_products)}  # +1 for padding
    idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}
    num_products = len(product_to_idx) + 1  # +1 for padding
    
    # Convert sessions to indices
    indexed_sessions = []
    for session in sessions:
        indexed_sessions.append([product_to_idx[pid] for pid in session])
    
    print(f"Number of unique products: {num_products}")
    
    return indexed_sessions, temporal_features, product_to_idx, idx_to_product, num_products

def split_data(sessions, temporal_features, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets"""
    print("Splitting data...")
    
    # First split: (train+val) and test
    train_val_sessions, test_sessions, train_val_temp, test_temp = train_test_split(
        sessions, temporal_features, test_size=test_size, random_state=SEED
    )
    
    # Second split: train and validation
    val_ratio = val_size / (1 - test_size)
    train_sessions, val_sessions, train_temp, val_temp = train_test_split(
        train_val_sessions, train_val_temp, test_size=val_ratio, random_state=SEED
    )
    
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Validation sessions: {len(val_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")
    
    return train_sessions, train_temp, val_sessions, val_temp, test_sessions, test_temp

def create_time_segments(test_sessions, test_temp):
    """Create time segments for analysis (morning/evening, weekday/weekend)"""
    
    # Morning vs Evening
    morning_indices = [i for i, temp in enumerate(test_temp) if temp['order_hour_of_day'] < 12]
    evening_indices = [i for i, temp in enumerate(test_temp) if temp['order_hour_of_day'] >= 12]
    
    morning_sessions = [test_sessions[i] for i in morning_indices]
    morning_temp = [test_temp[i] for i in morning_indices]
    
    evening_sessions = [test_sessions[i] for i in evening_indices]
    evening_temp = [test_temp[i] for i in evening_indices]
    
    # Weekday vs Weekend
    # 0 = Monday, ..., 6 = Sunday in Instacart dataset
    weekday_indices = [i for i, temp in enumerate(test_temp) if temp['order_dow'] < 5]  # Mon-Fri
    weekend_indices = [i for i, temp in enumerate(test_temp) if temp['order_dow'] >= 5]  # Sat-Sun
    
    weekday_sessions = [test_sessions[i] for i in weekday_indices]
    weekday_temp = [test_temp[i] for i in weekday_indices]
    
    weekend_sessions = [test_sessions[i] for i in weekend_indices]
    weekend_temp = [test_temp[i] for i in weekend_indices]
    
    # Cold-start (short) sessions
    cold_indices = [i for i, session in enumerate(test_sessions) if len(session) <= 3]
    cold_sessions = [test_sessions[i] for i in cold_indices]
    cold_temp = [test_temp[i] for i in cold_indices]
    
    time_segments = {
        'morning': (morning_sessions, morning_temp),
        'evening': (evening_sessions, evening_temp),
        'weekday': (weekday_sessions, weekday_temp),
        'weekend': (weekend_sessions, weekend_temp),
        'cold_start': (cold_sessions, cold_temp)
    }
    
    # Print statistics
    for segment, (sessions, _) in time_segments.items():
        print(f"{segment}: {len(sessions)} sessions")
    
    return time_segments

# Dataset class for training the models
class SessionDataset(Dataset):
    def __init__(self, sessions, temporal_features, max_len=20, is_test=False):
        self.sessions = sessions
        self.temporal_features = temporal_features
        self.max_len = max_len
        self.is_test = is_test
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        temp_features = self.temporal_features[idx]
        
        # For training/validation: use all but last item as input, last item as target
        if not self.is_test:
            # Handle very short sessions (need at least 2 items)
            if len(session) < 2:
                # Create a dummy session with padding and first item
                x = [0] * (self.max_len - 2) + [0, session[0]]
                y = session[0]  # Use the same item as target (not ideal but avoids errors)
                seq_len = 2
            else:
                x = session[:-1]
                y = session[-1]
                
                # Pad if needed
                seq_len = len(x)
                if seq_len < self.max_len - 1:
                    x = [0] * (self.max_len - 1 - seq_len) + x
                elif seq_len > self.max_len - 1:
                    x = x[-(self.max_len-1):]
                    seq_len = self.max_len - 1
                
            # Extract temporal features
            hour = temp_features['order_hour_of_day']
            dow = temp_features['order_dow']
            days_since = temp_features['days_since_prior_order']
            hour_sin = temp_features['hour_sin']
            hour_cos = temp_features['hour_cos']
            dow_sin = temp_features['dow_sin']
            dow_cos = temp_features['dow_cos']
            
            # Return with temporal features
            return {
                'input_ids': torch.tensor(x, dtype=torch.long),
                'attention_mask': torch.tensor([0] * (self.max_len - 1 - seq_len) + [1] * seq_len, dtype=torch.long),
                'target': torch.tensor(y, dtype=torch.long),
                'hour': torch.tensor(hour, dtype=torch.float),
                'dow': torch.tensor(dow, dtype=torch.float),
                'days_since': torch.tensor(days_since, dtype=torch.float),
                'hour_sin': torch.tensor(hour_sin, dtype=torch.float),
                'hour_cos': torch.tensor(hour_cos, dtype=torch.float),
                'dow_sin': torch.tensor(dow_sin, dtype=torch.float),
                'dow_cos': torch.tensor(dow_cos, dtype=torch.float),
            }
            
        # For testing: use all items as input, last item as target for evaluation
        else:
            # Handle very short sessions (need at least 2 items)
            if len(session) < 2:
                # Create a dummy session with padding and first item
                x = [0] * (self.max_len - 2) + [0, session[0]]
                y = session[0]  # Use the same item as target (not ideal but avoids errors)
                full_session = session.copy()
                seq_len = 2
            else:
                x = session[:-1]  # Input
                y = session[-1]   # Target
                full_session = session.copy()
                
                # Pad if needed
                seq_len = len(x)
                if seq_len < self.max_len - 1:
                    x = [0] * (self.max_len - 1 - seq_len) + x
                elif seq_len > self.max_len - 1:
                    x = x[-(self.max_len-1):]
                    seq_len = self.max_len - 1
                
            # Extract temporal features
            hour = temp_features['order_hour_of_day']
            dow = temp_features['order_dow']
            days_since = temp_features['days_since_prior_order']
            hour_sin = temp_features['hour_sin']
            hour_cos = temp_features['hour_cos']
            dow_sin = temp_features['dow_sin']
            dow_cos = temp_features['dow_cos']
            
            # Return with temporal features and full session for evaluation
            return {
                'input_ids': torch.tensor(x, dtype=torch.long),
                'attention_mask': torch.tensor([0] * (self.max_len - 1 - seq_len) + [1] * seq_len, dtype=torch.long),
                'target': torch.tensor(y, dtype=torch.long),
                'hour': torch.tensor(hour, dtype=torch.float),
                'dow': torch.tensor(dow, dtype=torch.float),
                'days_since': torch.tensor(days_since, dtype=torch.float),
                'hour_sin': torch.tensor(hour_sin, dtype=torch.float),
                'hour_cos': torch.tensor(hour_cos, dtype=torch.float),
                'dow_sin': torch.tensor(dow_sin, dtype=torch.float),
                'dow_cos': torch.tensor(dow_cos, dtype=torch.float),
                'full_session': full_session
            }

# Model architectures
class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (baseline model)
    Only uses item sequences without temporal information.
    """
    def __init__(self, num_items, hidden_size, num_layers, num_heads, dropout=0.1):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        
        # Item embedding
        self.item_embeddings = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(50, hidden_size)  # Max sequence length = 50
        self.dropout = nn.Dropout(dropout)
        
        # Multi-head attention layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_items)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, **kwargs):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        item_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = item_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for transformer (1 = attend, 0 = ignore)
        transformer_attn_mask = (1 - attention_mask).bool()
        
        # Transform through encoder
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=transformer_attn_mask)
        
        # Get last item representation
        sequence_output = encoded[:, -1, :]
        
        # Get logits
        logits = self.output_layer(sequence_output)
        
        return logits
    
    def get_loss(self, logits, target):
        return F.cross_entropy(logits, target)
    
    def predict(self, input_ids, attention_mask, **kwargs):
        logits = self.forward(input_ids, attention_mask, **kwargs)
        return logits

class TimeSASRec(nn.Module):
    """
    Time-Aware Self-Attentive Sequential Recommendation
    Extends SASRec with temporal features.
    """
    def __init__(self, num_items, hidden_size, num_layers, num_heads, dropout=0.1):
        super(TimeSASRec, self).__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        
        # Item embedding
        self.item_embeddings = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(50, hidden_size)  # Max sequence length = 50
        
        # Time embeddings
        self.hour_embeddings = nn.Linear(1, hidden_size)
        self.dow_embeddings = nn.Linear(1, hidden_size)
        self.days_since_embeddings = nn.Linear(1, hidden_size)
        
        # Cyclic time embeddings
        self.hour_cyclic_embeddings = nn.Linear(2, hidden_size)  # sin/cos
        self.dow_cyclic_embeddings = nn.Linear(2, hidden_size)   # sin/cos
        
        self.dropout = nn.Dropout(dropout)
        
        # Time gate mechanism for merging temporal and item embeddings
        self.time_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # Multi-head attention layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_items)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, hour, dow, days_since, 
                hour_sin, hour_cos, dow_sin, dow_cos, **kwargs):
        seq_length = input_ids.size(1)
        batch_size = input_ids.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get item and position embeddings
        item_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Process temporal features
        hour = hour.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
        dow = dow.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
        days_since = days_since.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
        
        # Process cyclic features
        hour_cyclic = torch.cat([
            hour_sin.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1),
            hour_cos.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
        ], dim=2)
        
        dow_cyclic = torch.cat([
            dow_sin.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1),
            dow_cos.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
        ], dim=2)
        
        # Get temporal embeddings
        hour_emb = self.hour_embeddings(hour)
        dow_emb = self.dow_embeddings(dow)
        days_since_emb = self.days_since_embeddings(days_since)
        hour_cyclic_emb = self.hour_cyclic_embeddings(hour_cyclic)
        dow_cyclic_emb = self.dow_cyclic_embeddings(dow_cyclic)
        
        # Combine temporal embeddings
        time_embeddings = hour_emb + dow_emb + days_since_emb + hour_cyclic_emb + dow_cyclic_emb
        
        # Gate mechanism for combining item and time embeddings
        gate_input = torch.cat([item_embeddings, time_embeddings], dim=-1)
        gate = torch.sigmoid(self.time_gate(gate_input))
        
        # Apply gate to combine embeddings
        combined_embeddings = item_embeddings * gate + time_embeddings * (1 - gate)
        
        # Add position embeddings and apply dropout
        embeddings = combined_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for transformer (1 = attend, 0 = ignore)
        transformer_attn_mask = (1 - attention_mask).bool()
        
        # Transform through encoder
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=transformer_attn_mask)
        
        # Get last item representation
        sequence_output = encoded[:, -1, :]
        
        # Get logits
        logits = self.output_layer(sequence_output)
        
        return logits
    
    def get_loss(self, logits, target):
        return F.cross_entropy(logits, target)
    
    def predict(self, input_ids, attention_mask, hour, dow, days_since, 
                hour_sin, hour_cos, dow_sin, dow_cos, **kwargs):
        logits = self.forward(
            input_ids, attention_mask, hour, dow, days_since,
            hour_sin, hour_cos, dow_sin, dow_cos, **kwargs
        )
        return logits

# Training and evaluation functions
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            logits = model(**batch)
            loss = model.get_loss(logits, batch['target'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, k_values=[5, 10, 20]):
    model.eval()
    metrics = {f'recall@{k}': 0.0 for k in k_values}
    metrics.update({f'ndcg@{k}': 0.0 for k in k_values})
    metrics.update({f'hit@{k}': 0.0 for k in k_values})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'full_session'}
            
            # Forward pass
            logits = model.predict(**batch)
            
            # Get predictions (exclude padding item)
            logits[:, 0] = -float('inf')  # Exclude padding item
            targets = batch['target'].unsqueeze(1)  # Shape: [batch_size, 1]
            
            # Calculate metrics for each k
            for k in k_values:
                # Get top-k predictions
                _, topk_indices = torch.topk(logits, k=k, dim=1)
                
                # Hit@k: whether target is in top-k
                hit = (topk_indices == targets).any(dim=1).float()
                metrics[f'hit@{k}'] += hit.sum().item()
                
                # Recall@k: 1 if target in top-k, 0 otherwise (same as hit for single target)
                metrics[f'recall@{k}'] += hit.sum().item()
                
                # NDCG@k: normalized discounted cumulative gain
                # Find position of target in top-k, if it exists
                target_pos = torch.nonzero(topk_indices == targets, as_tuple=True)[1]
                ndcg = torch.zeros_like(hit)
                ndcg[hit.bool()] = 1 / torch.log2(target_pos.float() + 2)  # +2 because positions are 0-indexed
                metrics[f'ndcg@{k}'] += ndcg.sum().item()
    
    # Normalize metrics
    num_samples = len(dataloader.dataset)
    for metric in metrics:
        metrics[metric] /= num_samples
        
    return metrics

def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, device, 
                      num_epochs=10, patience=3, model_name='model'):
    best_val_metric = 0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metric_dict = evaluate(model, val_loader, device)
        val_metric = val_metric_dict['recall@10']  # Use recall@10 as the main metric
        val_metrics.append(val_metric)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Recall@10: {val_metric:.4f}")
        
        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_name}_best.pth"))
            print(f"New best model saved with Recall@10: {val_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs, best epoch: {best_epoch+1}")
                break
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{model_name}_best.pth")))
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device)
    
    print("\nTest metrics:")
    for metric, value in sorted(test_metrics.items()):
        print(f"{metric}: {value:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_metrics)
    plt.title('Validation Recall@10')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@10')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_training_curve.png"))
    
    return test_metrics

def evaluate_time_segments(model, time_segments, model_name, device):
    """Evaluate model on different time segments"""
    results = {}
    
    for segment_name, (sessions, temp) in time_segments.items():
        print(f"\nEvaluating on {segment_name} segment...")
        
        # Create dataset and dataloader
        segment_dataset = SessionDataset(sessions, temp, is_test=True)
        segment_loader = DataLoader(segment_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
        
        # Evaluate
        segment_metrics = evaluate(model, segment_loader, device)
        results[segment_name] = segment_metrics
    
    # Save results
    result_df = pd.DataFrame()
    for segment, metrics in results.items():
        segment_df = pd.DataFrame([metrics])
        segment_df['segment'] = segment
        result_df = pd.concat([result_df, segment_df], ignore_index=True)
    
    result_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_segment_results.csv"), index=False)
    
    # Visualize results
    plt.figure(figsize=(15, 6))
    
    metrics_to_plot = ['recall@10', 'ndcg@10', 'hit@10']
    segments = result_df['segment'].tolist()
    
    x = np.arange(len(segments))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = result_df[metric].tolist()
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Time Segment')
    plt.ylabel('Score')
    plt.title(f'{model_name} Performance Across Time Segments')
    plt.xticks(x + width, segments, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_segment_performance.png"))
    
    return results

def ablation_study(train_sessions, train_temp, val_sessions, val_temp, test_sessions, test_temp, 
                  num_products, device, base_config):
    """
    Perform ablation study by removing each time feature one at a time
    """
    print("\n=== Starting Ablation Study ===")
    
    # Define features to ablate
    ablation_configs = {
        'all_features': {  # baseline with all features
            'use_hour': True,
            'use_dow': True, 
            'use_days_since': True,
            'use_cyclic': True
        },
        'no_hour': {
            'use_hour': False,
            'use_dow': True, 
            'use_days_since': True,
            'use_cyclic': True
        },
        'no_dow': {
            'use_hour': True,
            'use_dow': False, 
            'use_days_since': True,
            'use_cyclic': True
        },
        'no_days_since': {
            'use_hour': True,
            'use_dow': True, 
            'use_days_since': False,
            'use_cyclic': True
        },
        'no_cyclic': {
            'use_hour': True,
            'use_dow': True, 
            'use_days_since': True,
            'use_cyclic': False
        }
    }
    
    results = {}
    
    # Modified TimeSASRec class for ablation
    class AblationTimeSASRec(TimeSASRec):
        def __init__(self, num_items, hidden_size, num_layers, num_heads, dropout=0.1,
                    use_hour=True, use_dow=True, use_days_since=True, use_cyclic=True):
            super(AblationTimeSASRec, self).__init__(num_items, hidden_size, num_layers, num_heads, dropout)
            self.use_hour = use_hour
            self.use_dow = use_dow
            self.use_days_since = use_days_since
            self.use_cyclic = use_cyclic
            
        def forward(self, input_ids, attention_mask, hour, dow, days_since, 
                   hour_sin, hour_cos, dow_sin, dow_cos, **kwargs):
            seq_length = input_ids.size(1)
            batch_size = input_ids.size(0)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
            # Get item and position embeddings
            item_embeddings = self.item_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            
            # Initialize temporal embeddings
            time_embeddings = torch.zeros_like(item_embeddings)
            
            # Conditionally add temporal features based on ablation configuration
            if self.use_hour:
                hour = hour.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
                hour_emb = self.hour_embeddings(hour)
                time_embeddings = time_embeddings + hour_emb
            
            if self.use_dow:
                dow = dow.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
                dow_emb = self.dow_embeddings(dow)
                time_embeddings = time_embeddings + dow_emb
            
            if self.use_days_since:
                days_since = days_since.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
                days_since_emb = self.days_since_embeddings(days_since)
                time_embeddings = time_embeddings + days_since_emb
            
            if self.use_cyclic:
                # Process cyclic features
                hour_cyclic = torch.cat([
                    hour_sin.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1),
                    hour_cos.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
                ], dim=2)
                
                dow_cyclic = torch.cat([
                    dow_sin.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1),
                    dow_cos.unsqueeze(1).unsqueeze(2).expand(-1, seq_length, 1)
                ], dim=2)
                
                hour_cyclic_emb = self.hour_cyclic_embeddings(hour_cyclic)
                dow_cyclic_emb = self.dow_cyclic_embeddings(dow_cyclic)
                
                time_embeddings = time_embeddings + hour_cyclic_emb + dow_cyclic_emb
            
            # Gate mechanism for combining item and time embeddings
            gate_input = torch.cat([item_embeddings, time_embeddings], dim=-1)
            gate = torch.sigmoid(self.time_gate(gate_input))
            
            # Apply gate to combine embeddings
            combined_embeddings = item_embeddings * gate + time_embeddings * (1 - gate)
            
            # Add position embeddings and apply dropout
            embeddings = combined_embeddings + position_embeddings
            embeddings = self.dropout(embeddings)
            
            # Create attention mask for transformer
            transformer_attn_mask = (1 - attention_mask).bool()
            
            # Transform through encoder
            encoded = self.transformer_encoder(embeddings, src_key_padding_mask=transformer_attn_mask)
            
            # Get last item representation
            sequence_output = encoded[:, -1, :]
            
            # Get logits
            logits = self.output_layer(sequence_output)
            
            return logits
    
    # Create dataloaders
    # Create dataloaders
    train_dataset = SessionDataset(train_sessions, train_temp)
    val_dataset = SessionDataset(val_sessions, val_temp, is_test=True)
    test_dataset = SessionDataset(test_sessions, test_temp, is_test=True)
    
    # Batch size for full dataset
    batch_size = 32  # Reduced for memory efficiency
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    for ablation_name, config in ablation_configs.items():
        print(f"\nTraining ablation model: {ablation_name}")
        
        # Create model with specific ablation configuration
        model = AblationTimeSASRec(
            num_items=num_products,
            hidden_size=base_config['hidden_size'],
            num_layers=base_config['num_layers'],
            num_heads=base_config['num_heads'],
            dropout=base_config['dropout'],
            use_hour=config['use_hour'],
            use_dow=config['use_dow'],
            use_days_since=config['use_days_since'],
            use_cyclic=config['use_cyclic']
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=base_config['learning_rate'])
        
        # Train and evaluate
        test_metrics = train_and_evaluate(
            model, train_loader, val_loader, test_loader, optimizer, device,
            num_epochs=base_config['num_epochs'], 
            patience=base_config['patience'],
            model_name=f"ablation_{ablation_name}"
        )
        
        results[ablation_name] = test_metrics
    
    # Summarize ablation results
    ablation_df = pd.DataFrame()
    for ablation_name, metrics in results.items():
        ablation_df_row = pd.DataFrame([metrics])
        ablation_df_row['ablation'] = ablation_name
        ablation_df = pd.concat([ablation_df, ablation_df_row], ignore_index=True)
    
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)
    
    # Visualize ablation results
    plt.figure(figsize=(15, 6))
    
    metrics_to_plot = ['recall@10', 'ndcg@10', 'hit@10']
    ablations = ablation_df['ablation'].tolist()
    
    x = np.arange(len(ablations))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = ablation_df[metric].tolist()
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Ablation Configuration')
    plt.ylabel('Score')
    plt.title('Ablation Study Results')
    plt.xticks(x + width, ablations, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_study_results.png"))
    
    return results

def custom_collate_fn(batch):
    """Custom collate function that handles variable length sessions"""
    elem = batch[0]
    result = {}
    
    # Handle all keys except 'full_session' with default_collate
    for key in elem:
        if key != 'full_session':
            result[key] = torch.stack([item[key] for item in batch])
    
    # Handle 'full_session' separately if it exists
    if 'full_session' in elem:
        result['full_session'] = [item['full_session'] for item in batch]
        
    return result

# Main execution function
def main():
    print("Starting time-aware recommender system experiment...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    orders, order_products_prior, products, aisles, departments = load_data()
    
    # Sample a subset of data for quick experimentation (remove for full experiment)
    # orders = orders.sample(frac=0.1, random_state=SEED).reset_index(drop=True)
    # order_ids = orders['order_id'].unique()
    # order_products_prior = order_products_prior[order_products_prior['order_id'].isin(order_ids)]
    
    # Preprocess data
    sessions, temporal_features, product_to_idx, idx_to_product, num_products = preprocess_data(
        orders, order_products_prior, products
    )
    
    # Split data
    train_sessions, train_temp, val_sessions, val_temp, test_sessions, test_temp = split_data(
        sessions, temporal_features
    )
    
    # Create time segments for analysis
    time_segments = create_time_segments(test_sessions, test_temp)
    
    # Create datasets and dataloaders
    train_dataset = SessionDataset(train_sessions, train_temp)
    val_dataset = SessionDataset(val_sessions, val_temp, is_test=True)
    test_dataset = SessionDataset(test_sessions, test_temp, is_test=True)
    
    # Batch size for full dataset
    batch_size = 32  # Reduced from 128 for memory efficiency with full dataset
    
    # Use custom_collate_fn with the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Model configuration
    config = {
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'patience': 5
    }
    
    # Train baseline model (SASRec)
    print("\n=== Training Baseline Model (SASRec) ===")
    baseline_model = SASRec(
        num_items=num_products,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=config['learning_rate'])
    
    baseline_metrics = train_and_evaluate(
        baseline_model, train_loader, val_loader, test_loader, baseline_optimizer, device,
        num_epochs=config['num_epochs'], patience=config['patience'], model_name="sasrec_baseline"
    )
    
    # Evaluate baseline on time segments
    baseline_segment_results = evaluate_time_segments(
        baseline_model, time_segments, "sasrec_baseline", device
    )
    
    # Train time-aware model (TimeSASRec)
    print("\n=== Training Time-Aware Model (TimeSASRec) ===")
    time_model = TimeSASRec(
        num_items=num_products,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    
    time_optimizer = torch.optim.Adam(time_model.parameters(), lr=config['learning_rate'])
    
    time_metrics = train_and_evaluate(
        time_model, train_loader, val_loader, test_loader, time_optimizer, device,
        num_epochs=config['num_epochs'], patience=config['patience'], model_name="timesasrec"
    )
    
    # Evaluate time-aware model on time segments
    time_segment_results = evaluate_time_segments(
        time_model, time_segments, "timesasrec", device
    )
    
    # Perform ablation study
    ablation_results = ablation_study(
        train_sessions, train_temp, val_sessions, val_temp, test_sessions, test_temp,
        num_products, device, config
    )
    
    # Compare baseline vs time-aware model
    comparison_df = pd.DataFrame([
        {'model': 'SASRec (Baseline)', **baseline_metrics},
        {'model': 'TimeSASRec', **time_metrics}
    ])
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    
    # Visualize comparison
    metrics_to_plot = ['recall@10', 'ndcg@10', 'hit@10']
    models = comparison_df['model'].tolist()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = comparison_df[metric].tolist()
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('SASRec vs. TimeSASRec Performance')
    plt.xticks(x + width, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    
    print("\nExperiment completed.")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()