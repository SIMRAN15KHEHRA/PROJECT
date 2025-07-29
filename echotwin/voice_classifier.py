"""
Voice Disorder Classification Module for EchoTwin

This module implements multiple deep learning models for voice disorder detection:
- CNN-based classifier for spectral features
- LSTM-based classifier for temporal features
- Transformer-based classifier for sequence modeling
- Ensemble methods combining multiple approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class VoiceDataset(Dataset):
    """
    Custom dataset for voice features and labels.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize dataset.
        
        Args:
            features: Feature vectors
            labels: Corresponding labels
            transform: Optional transform to apply to features
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label


class CNN1DClassifier(nn.Module):
    """
    1D CNN for voice disorder classification.
    """
    
    def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.5):
        """
        Initialize CNN classifier.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(CNN1DClassifier, self).__init__()
        
        # Reshape input for 1D convolution
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Reshape for 1D convolution (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for voice disorder detection.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout_rate: float = 0.5):
        """
        Initialize LSTM classifier.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # For single feature vector, create sequence of length 1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, seq_len=1, features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(attended_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for voice disorder detection.
    """
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, num_classes: int, dropout_rate: float = 0.1):
        """
        Initialize Transformer classifier.
        
        Args:
            input_size: Number of input features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # For single feature vector, create sequence of length 1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, seq_len=1, features)
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class VoiceClassifier:
    """
    Main voice disorder classifier with multiple model options.
    """
    
    def __init__(self, model_type: str = 'cnn', device: str = 'auto'):
        """
        Initialize voice classifier.
        
        Args:
            model_type: Type of model ('cnn', 'lstm', 'transformer', 'ensemble')
            device: Device to use ('cuda', 'cpu', 'auto')
        """
        self.model_type = model_type
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.num_classes = None
        self.trained = False
        
    def prepare_data(self, features: np.ndarray, labels: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            features: Feature matrix
            labels: Labels
            test_size: Proportion of test data
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(np.unique(y_encoded))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_model(self, input_size: int) -> nn.Module:
        """
        Create model based on specified type.
        
        Args:
            input_size: Number of input features
            
        Returns:
            PyTorch model
        """
        if self.model_type == 'cnn':
            model = CNN1DClassifier(input_size, self.num_classes)
        elif self.model_type == 'lstm':
            model = LSTMClassifier(input_size, hidden_size=128, num_layers=2, 
                                 num_classes=self.num_classes)
        elif self.model_type == 'transformer':
            model = TransformerClassifier(input_size, d_model=128, nhead=8, 
                                        num_layers=4, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
              patience: int = 10, verbose: bool = True) -> Dict:
        """
        Train the classifier.
        
        Args:
            features: Feature matrix
            labels: Labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features, labels)
        
        # Create datasets
        train_dataset = VoiceDataset(X_train, y_train)
        test_dataset = VoiceDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        self.model = self.create_model(X_train.shape[1])
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(test_loader))
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {history["train_loss"][-1]:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {history["val_loss"][-1]:.4f}, Val Acc: {val_acc:.2f}%')
                print('-' * 50)
            
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.trained = True
        
        if verbose:
            print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        return history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predicted.cpu().numpy())
        
        return predicted_labels
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            features: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            features: Feature matrix
            labels: True labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'num_classes': self.num_classes,
            'input_size': list(self.model.parameters())[0].shape[1] if self.model_type == 'cnn' else list(self.model.parameters())[0].shape[0]
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        self.num_classes = checkpoint['num_classes']
        
        # Create model
        input_size = checkpoint['input_size']
        self.model = self.create_model(input_size)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = True
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, history: Dict) -> None:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
        """
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = self.label_encoder.classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    
    # Create synthetic features (simulating extracted voice features)
    n_samples = 1000
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels (healthy vs. disorder)
    y = np.random.choice(['healthy', 'disorder'], n_samples)
    
    print("Testing VoiceClassifier...")
    
    # Test CNN classifier
    print("Testing CNN classifier...")
    cnn_classifier = VoiceClassifier(model_type='cnn')
    history = cnn_classifier.train(X, y, epochs=20, verbose=False)
    
    # Evaluate
    results = cnn_classifier.evaluate(X, y)
    print(f"CNN Accuracy: {results['accuracy']:.3f}")
    
    # Test LSTM classifier
    print("Testing LSTM classifier...")
    lstm_classifier = VoiceClassifier(model_type='lstm')
    history = lstm_classifier.train(X, y, epochs=20, verbose=False)
    
    # Evaluate
    results = lstm_classifier.evaluate(X, y)
    print(f"LSTM Accuracy: {results['accuracy']:.3f}")
    
    print("VoiceClassifier test completed successfully!")