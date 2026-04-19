


# this is the model training code which I have used to show in the lab
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
import os

# --- 1. LOAD PROCESSED DATA ---
DATA_PATH = r"D:\BE. CSE\Sem 6\IoT_lab\BabyCry_v2_Pro\data\processed"
X = np.load(os.path.join(DATA_PATH, "X_data.npy"))
y = np.load(os.path.join(DATA_PATH, "y_data.npy"))

print(f"--- Dataset Loaded ---")
print(f"Total Samples: {X.shape[0]}")

# --- 2. SPLIT DATA ---
# Using stratify=y ensures both Train and Test have an equal % of Crying vs Normal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. THE STABILIZED CNN ARCHITECTURE ---
model = models.Sequential([
    # Layer 1: Edge Detection + L2 Regularization
    layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=(128, 128, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 2: Frequency Patterns
    layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 3: Complex Textures
    layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Global Average Pooling (Alternative to Flatten to reduce parameters/overfitting)
    layers.GlobalAveragePooling2D(),
    
    # Fully Connected
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(1, activation='sigmoid')
])

# Use a slightly lower initial learning rate for stability
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# --- 4. CALLBACKS ---
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, # Increased patience to allow for ReduceLROnPlateau to work
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, # Reduce by half
    patience=3, 
    min_lr=0.00001,
    verbose=1
)

# --- 5. START TRAINING ---
print("\n🚀 Starting Stabilized Training...")
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    shuffle=True, # Critical for stability
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# --- 6. SAVE AND CONVERT ---
if not os.path.exists('models'):
    os.makedirs('models')

# Save Keras Model
h5_path = "models/baby_cry_v2_pro.h5"
model.save(h5_path)
print(f"\n✅ Full Model saved: {h5_path}")

# Convert to TFLite (Optimized for Backend)
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = "models/baby_cry_v2_pro.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"🎯 TFLite Model saved: {tflite_path}")

