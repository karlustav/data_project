import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# Load and preprocess data
data = pd.read_csv('../../data/preprocessed_data.csv')  # Adjusted path
X = data.drop('Depression', axis=1)
y = data['Depression']

# Split and scale data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Add data augmentation
def augment_data(X, y, noise_scale=0.05):
    noise = np.random.normal(0, noise_scale, X.shape)
    X_augmented = X + noise
    return np.vstack([X, X_augmented]), np.concatenate([y, y])

# Augment training data
X_train_aug, y_train_aug = augment_data(X_train_scaled, y_train)

# Convert to numpy arrays and calculate class weights
y_train_aug = np.array(y_train_aug)
class_counts = np.bincount(y_train_aug.astype(int))
total_samples = len(y_train_aug)

class_weights = {
    0: total_samples / (2 * class_counts[0]),
    1: total_samples / (2 * class_counts[1])
}

# Create an enhanced model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.GaussianNoise(0.05),  # Reduced noise
    
    # First block - highest capacity
    layers.Dense(1024, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),
                kernel_initializer='he_uniform'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Second block
    layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),
                kernel_initializer='he_uniform'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    # Third block
    layers.Dense(256, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),
                kernel_initializer='he_uniform'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Fourth block
    layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),
                kernel_initializer='he_uniform'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Fifth block
    layers.Dense(64, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),
                kernel_initializer='he_uniform'),
    layers.LeakyReLU(negative_slope=0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    
    # Output layer with stronger regularization
    layers.Dense(1, activation='sigmoid',
                kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))
])

# Enhanced callbacks
early_stopping = EarlyStopping(
    monitor='val_auc',  # Changed to AUC for better monitoring
    patience=25,
    restore_best_weights=True,
    mode='max',
    min_delta=0.001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.2,
    patience=10,
    min_lr=1e-7,
    mode='max',
    verbose=1
)

# Cosine decay learning rate schedule
initial_learning_rate = 0.001
decay_steps = 1000
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps, alpha=1e-6
)

# Enhanced optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    amsgrad=True
)

# Compile with additional metrics
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

# Train with augmented data
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val_scaled, y_val),
    epochs=500,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Add model evaluation
test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")