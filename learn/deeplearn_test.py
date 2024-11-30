import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

# Step 1: Load your dataset
# Replace with your dataset
data = pd.read_csv('preprocessed_data.csv')

# Step 2: Preprocess the dataset
# Separate features (X) and target (y)
X = data.drop('Depresseion', axis=1).values  # Replace 'target_column' with your target column name
y = data['Depression']

# Encode categorical target (if needed)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Helps prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')  # Adjust output layer for the number of classes
])

# Step 4: Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use binary_crossentropy for binary classification
    metrics=['accuracy']
)

# Step 5: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Number of training cycles
    batch_size=32,
    verbose=1
)

# Step 6: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 7: Save the model
model.save('my_model.h5')

# Step 8: Optional - Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
