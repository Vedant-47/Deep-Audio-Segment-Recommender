"""
Simple backend: load GTZAN-like folder structure:
root/
    blues/
      track1.mp3
      ...
    classical/
      ...
Train a small CNN on Mel-spectrograms, save:
 - keras model (model.h5)
 - embeddings.npy (N x D)
 - metadata.csv (id, filepath, genre)
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import io
# import warnings
# warnings.filterwarnings('ignore')

# -------- CONFIG --------
DATA_ROOT = "gtzan"      # path to root folder containing genre subfolders
TARGET_SR = 22050
DURATION = 30           # seconds
N_MELS = 128
IMG_SHAPE = (128, 128)  # we'll resize mel spectrogram to this (freq x time)
BATCH_SIZE = 16
EPOCHS = 100
EMBEDDING_DIM = 128
OUT_DIR = "model_output"
os.makedirs(OUT_DIR, exist_ok=True)


# -------- HELPERS --------
def load_audio(path, duration=DURATION, sr=TARGET_SR):
    y, sr = librosa.load(path, sr=sr, duration=duration, mono=True)
    if len(y) < sr * duration:
        # pad
        y = np.pad(y, (0, max(0, sr*duration - len(y))), mode='constant')
    return y, sr

def mel_to_image(y, sr, n_mels=N_MELS, shape=IMG_SHAPE):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    # normalize to [0,1]
    Smin, Smax = S_db.min(), S_db.max()
    S_scaled = (S_db - Smin) / (Smax - Smin + 1e-6)
    # resize to shape (simple way: using numpy interpolation)
    img = np.array(
        plt.imshow(S_scaled, aspect='auto').get_array()
    )
    plt.clf()
    # but above method may not be consistent cross-platform; instead use resizing via numpy:
    img = np.array(S_scaled)
    img = np.flip(img, axis=0)  # put low freqs bottom
    # resize time dimension
    from skimage.transform import resize
    img_resized = resize(img, shape, mode='reflect', anti_aliasing=True)
    return img_resized.astype(np.float32)


# -------- BUILD DATASET --------
print("Scanning dataset...")
rows = []
genres = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
for genre in genres:
    folder = os.path.join(DATA_ROOT, genre)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.mp3', '.wav', '.au')):
            rows.append({
                "filepath": os.path.join(folder, fname),
                "genre": genre
            })
df = pd.DataFrame(rows)
df['id'] = df.index.astype(str)
print(f"Found {len(df)} tracks across {len(genres)} genres.")

# For speed during development you may want to sample fewer tracks:
# df = df.sample(n=200, random_state=1).reset_index(drop=True)

# -------- EXTRACT MEL IMAGES & LABELS --------
print("Extracting mel images (this will take time)...")
X = []
y = []
filepaths = []
for idx, r in df.iterrows():
    y_audio, sr = load_audio(r['filepath'])
    img = mel_to_image(y_audio, sr)
    X.append(img)
    y.append(genres.index(r['genre']))
    filepaths.append(r['filepath'])
X = np.stack(X)
y = np.array(y)
print("X shape:", X.shape, "y shape:", y.shape)

# add channel dimension
X = X[..., np.newaxis]

# -------- MODEL (small CNN) --------
def build_model(input_shape=(*IMG_SHAPE, 1), n_classes=len(genres), embedding_dim=EMBEDDING_DIM):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(embedding_dim, activation='relu', name='embedding')(x)
    out = layers.Dense(n_classes, activation='softmax')(embedding)
    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------- TRAIN --------
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
print("Training ...")
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

# -------- SAVE MODEL & EMBEDDINGS --------
# Save entire model
model_path = os.path.join(OUT_DIR, "genre_cnn.h5")
model.save(model_path)
print("Saved model:", model_path)

# Create an embedding extractor (model without final softmax)
embed_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('embedding').output)
# compute embeddings for all tracks
print("Computing embeddings...")
embeddings = embed_model.predict(X, batch_size=16)
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
print("Saved embeddings:", os.path.join(OUT_DIR, "embeddings.npy"))

# Save metadata
meta = pd.DataFrame({
    "id": df['id'].astype(str),
    "filepath": filepaths,
    "genre": [genres[i] for i in y]
})
meta.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)
print("Saved metadata.csv")
