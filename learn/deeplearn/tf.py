import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from tensorflow.keras.metrics import AUC, Precision, Recall, F1Score

# Load data
data = pd.read_csv('../../data/preprocessed_data.csv')
X = data.drop('Depression', axis=1)
y = data['Depression']

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights
class_counts = np.bincount(y_train_balanced.astype(int))
total_samples = len(y_train_balanced)
class_weights = {
    0: total_samples / (2 * class_counts[0]),
    1: total_samples / (2 * class_counts[1])
}

# Attention layer function
def attention_layer(inputs):
    attention_probs = layers.Dense(inputs.shape[-1], activation='softmax')(inputs)
    return layers.Multiply()([inputs, attention_probs])

# Model architecture
inputs = layers.Input(shape=(X_train.shape[1],))
x = layers.GaussianNoise(0.05)(inputs)

# First block with matching dimensions for residual connection
residual = layers.Dense(2048, kernel_initializer='he_uniform')(x)  # Projection for residual
x = layers.Dense(2048, kernel_initializer='he_uniform')(x)
x = layers.LeakyReLU(negative_slope=0.1)(x)
x = layers.BatchNormalization()(x)
x = attention_layer(x)
x = layers.Dropout(0.4)(x)
x = layers.Add()([residual, x])  # Now dimensions match

# Dense blocks
for units in [1024, 512, 256]:
    x = layers.Dense(units, kernel_initializer='he_uniform',
                    kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Learning rate schedule
initial_learning_rate = 0.001
decay_steps = len(X_train_scaled) // 64 * 10  # Update steps per epoch
lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate, first_decay_steps=decay_steps, alpha=1e-6
)

# Optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    amsgrad=True
)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy',
             AUC(name='auc'),
             Precision(name='precision'),
             Recall(name='recall'),
             F1Score(name='f1_score')]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_f1_score',  # Focus on F1 Score for imbalanced data
    patience=30,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_f1_score',
    factor=0.2,
    patience=10,
    min_lr=1e-7,
    mode='max',
    verbose=1
)

# Train the model
history = model.fit(
    X_train_scaled, y_train_balanced,
    validation_data=(X_val_scaled, y_val),
    epochs=300,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
test_results = model.evaluate(X_test_scaled, y_test, verbose=1)
print("\nTest Results:")
print(f"Accuracy: {test_results[1]:.4f}")
print(f"AUC: {test_results[2]:.4f}")
print(f"Precision: {test_results[3]:.4f}")
print(f"Recall: {test_results[4]:.4f}")
print(f"F1 Score: {test_results[5]:.4f}")
