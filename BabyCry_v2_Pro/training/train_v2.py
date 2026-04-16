# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras import layers, models
# # from sklearn.model_selection import train_test_split
# # import os

# # # --- 1. LOAD PROCESSED DATA ---
# # DATA_PATH = r"D:\BE. CSE\Sem 6\IoT_lab\BabyCry_v2_Pro\data\processed"
# # X = np.load(os.path.join(DATA_PATH, "X_data.npy"))
# # y = np.load(os.path.join(DATA_PATH, "y_data.npy"))

# # print(f"Loaded {X.shape[0]} samples.")

# # # --- 2. SPLIT DATA (80% Train, 20% Test) ---
# # # This ensures we have a "Final Exam" for the AI to prove it's well-fit
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # --- 3. DEFINE THE CNN ARCHITECTURE ---
# # model = models.Sequential([
# #     # Input is 128x128 Mel Spectrogram
# #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
# #     layers.MaxPooling2D((2, 2)),
    
# #     layers.Conv2D(64, (3, 3), activation='relu'),
# #     layers.MaxPooling2D((2, 2)),
    
# #     layers.Conv2D(128, (3, 3), activation='relu'),
# #     layers.MaxPooling2D((2, 2)),
    
# #     layers.Flatten(),
# #     layers.Dense(128, activation='relu'),
# #     layers.Dropout(0.5),  # Crucial: Helps ignore your specific room noise/voice
# #     layers.Dense(1, activation='sigmoid') # 0 = Noise/Speech, 1 = Crying
# # ])

# # model.compile(optimizer='adam', 
# #               loss='binary_crossentropy', 
# #               metrics=['accuracy'])

# # model.summary()

# # # --- 4. TRAIN THE MODEL ---
# # print("\nStarting Training...")
# # # Increase epochs to 30 for better fitting
# # history = model.fit(X_train, y_train, 
# #                     epochs=30, 
# #                     batch_size=32, 
# #                     validation_data=(X_test, y_test))

# # # --- 5. SAVE AS KERAS (.h5) ---
# # if not os.path.exists('models'):
# #     os.makedirs('models')

# # h5_path = "models/baby_cry_v2_pro.h5"
# # model.save(h5_path)
# # print(f"\n[SUCCESS] Model saved to {h5_path}")

# # # --- 6. CONVERT & SAVE AS TFLITE (.tflite) ---
# # # This is the optimized version for your FastAPI backend
# # print("Converting to TFLite...")
# # converter = tf.lite.TFLiteConverter.from_keras_model(model)
# # tflite_model = converter.convert()

# # tflite_path = "models/baby_cry_v2_pro.tflite"
# # with open(tflite_path, "wb") as f:
# #     f.write(tflite_model)

# # print(f"[SUCCESS] Optimized model saved to {tflite_path}")

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# from sklearn.model_selection import train_test_split
# import os

# # --- 1. LOAD PROCESSED DATA ---
# DATA_PATH = r"D:\BE. CSE\Sem 6\IoT_lab\BabyCry_v2_Pro\data\processed"
# X = np.load(os.path.join(DATA_PATH, "X_data.npy"))
# y = np.load(os.path.join(DATA_PATH, "y_data.npy"))

# print(f"--- Dataset Loaded ---")
# print(f"Total Samples: {X.shape[0]}")
# print(f"Input Shape: {X.shape[1:]}")

# # --- 2. SPLIT DATA (80% Train, 20% Test) ---
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # --- 3. DEFINE THE "WELL-FIT" CNN ARCHITECTURE ---
# model = models.Sequential([
#     # Layer 1: Edge Detection
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)),
#     layers.BatchNormalization(), # Keeps training stable
#     layers.MaxPooling2D((2, 2)),
    
#     # Layer 2: Frequency Pattern Detection
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
    
#     # Layer 3: Complex Texture Detection
#     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
    
#     # Flatten and Classify
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.6), # High dropout to prevent "memorizing" specific voices
#     layers.Dense(1, activation='sigmoid') # 0 = Noise/Speech, 1 = Baby Crying
# ])

# model.compile(optimizer='adam', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# # --- 4. SMART TRAINING CALLBACKS ---
# # Stops training if it's not getting better, saving the best version
# early_stop = callbacks.EarlyStopping(
#     monitor='val_loss', 
#     patience=7, 
#     restore_best_weights=True,
#     verbose=1
# )

# # Reduces learning rate if progress stalls (helps find a better "fit")
# reduce_lr = callbacks.ReduceLROnPlateau(
#     monitor='val_loss', 
#     factor=0.2, 
#     patience=3, 
#     min_lr=0.00001,
#     verbose=1
# )

# # --- 5. START TRAINING ---
# print("\nStarting Training...")
# history = model.fit(
#     X_train, y_train, 
#     epochs=50, 
#     batch_size=32, 
#     validation_data=(X_test, y_test),
#     callbacks=[early_stop, reduce_lr]
# )

# # --- 6. SAVE AND CONVERT ---
# if not os.path.exists('models'):
#     os.makedirs('models')

# # Save Keras Model
# h5_path = "models/baby_cry_v2_pro.h5"
# model.save(h5_path)
# print(f"\n[SUCCESS] Full Model saved: {h5_path}")

# # Convert to TFLite (Optimized for Backend)
# print("Converting to TFLite...")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# tflite_path = "models/baby_cry_v2_pro.tflite"
# with open(tflite_path, "wb") as f:
#     f.write(tflite_model)

# print(f"[SUCCESS] TFLite Model saved: {tflite_path}")



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

