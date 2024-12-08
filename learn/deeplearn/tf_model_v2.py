import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.python.keras import layers, regularizers
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

# Load the dataset
# Replace 'data_project/data/preprocessed_data.csv' with your file path
data = pd.read_csv('data_project/data/preprocessed_data.csv')
X = data.drop(columns=['Depression'])
y = data['Depression']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reset indices to align them properly
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Add data augmentation for numerical data
def augment_numerical_data(X, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

# Model architecture with regularization
def build_model(input_dim, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.GaussianNoise(0.15),
        
        # First dense block
        layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5), kernel_initializer='he_uniform'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second dense block
        layers.Dense(256, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5), kernel_initializer='he_uniform'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense block
        layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5), kernel_initializer='he_uniform'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(2, activation='softmax')
    ])
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Simpler metrics during debugging
    )
    return model

# Calculate class weights
def calculate_class_weights(y):
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    return weights

# Prepare data for training
X_train_augmented = augment_numerical_data(X_train_scaled)
X_train_combined = np.vstack([X_train_scaled, X_train_augmented])
y_train_combined = np.concatenate([y_train.to_numpy(), y_train.to_numpy()])

# Check data alignment
print("X_train_combined shape:", X_train_combined.shape)
print("y_train_combined shape:", y_train_combined.shape)
print("X_val_scaled shape:", X_val_scaled.shape)
print("y_val shape:", y_val.shape)

# Calculate class weights
class_weights = calculate_class_weights(y_train)

# Build the model
model = build_model(X_train_scaled.shape[1])
model.summary()

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',  # Use loss during debugging for simplicity
        patience=20,
        restore_best_weights=True,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
history = model.fit(
    X_train_combined,
    y_train_combined,
    validation_data=(X_val_scaled, y_val),
    epochs=300,
    batch_size=128,  # Ensure this aligns with data size
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Plot training history
def plot_training_history(history):
    metrics = ['loss', 'accuracy']
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        axes[idx].plot(history.history[metric], label=f'Train {metric}')
        axes[idx].plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        axes[idx].set_title(f'{metric.capitalize()} Over Epochs')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric)
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history(history)
