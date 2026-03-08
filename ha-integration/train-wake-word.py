#!/usr/bin/env python3
"""Train a custom 'hey cedric' wake word model for openWakeWord (HA addon).

Pipeline:
1. Generate positive WAV samples ("hey cedric" variations) using Piper TTS
2. Generate negative WAV samples (similar phrases, random speech)
3. Extract 96-dim embeddings using openWakeWord's embed_clips
4. Train a simple classifier on the embedding windows
5. Export to TFLite
"""

import os
import subprocess
import random
import struct
import numpy as np
from pathlib import Path

PIPER_BIN = "/opt/piper/piper"
PIPER_DIR = "/opt/piper"
MODEL_DIR = "/opt/piper/models"
OUTPUT_DIR = "/tmp/hey-cedric-training"
FINAL_MODEL = "/tmp/hey_cedric.tflite"

WW_FEATURES = 96
INPUT_WINDOWS = 16

VOICES = [
    (f"{MODEL_DIR}/en_GB-semaine-medium.onnx", "0"),
    (f"{MODEL_DIR}/en_GB-semaine-medium.onnx", "1"),
    (f"{MODEL_DIR}/en_GB-semaine-medium.onnx", "2"),
    (f"{MODEL_DIR}/en_GB-semaine-medium.onnx", "3"),
    (f"{MODEL_DIR}/en_GB-alan-medium.onnx", ""),
    (f"{MODEL_DIR}/en_US-amy-medium.onnx", ""),
    (f"{MODEL_DIR}/en_US-bryce-medium.onnx", ""),
    (f"{MODEL_DIR}/en_US-hfc_female-medium.onnx", ""),
    (f"{MODEL_DIR}/en_US-kristin-medium.onnx", ""),
    (f"{MODEL_DIR}/en_US-kusal-medium.onnx", ""),
    (f"{MODEL_DIR}/en_US-lessac-medium.onnx", ""),
]

POSITIVE_PHRASES = [
    "hey cedric", "Hey Cedric", "hey Cedric", "Hey cedric",
    "hey cedric!", "Hey Cedric!", "hey, cedric", "Hey, Cedric",
    "hey cedric.", "Hey Cedric?",
]

NEGATIVE_PHRASES = [
    "hey derek", "hey patrick", "hey frederick", "hey hendrick",
    "hey electric", "hey metric", "a cedric", "the cedric",
    "say cedric", "play cedric", "hey", "cedric",
    "good morning", "hello there", "hey siri", "hey google",
    "hey jarvis", "okay", "hey buddy", "hey there", "what's up",
    "hey everyone", "hey eric", "hey craig", "hey chris",
    "hey sergio", "hey derick", "hey dedric", "hey generic",
    "hey fabric", "the weather is nice", "turn on the lights",
    "what time is it", "play some music", "set a timer",
    "hey alexa", "okay google", "hey cortana", "hey mycroft",
    "open the door", "close the window", "hey netflix",
    "hey hendrix", "make some coffee",
]


def generate_wav(text, model_path, speaker, output_path, length_scale=None):
    if length_scale is None:
        length_scale = random.uniform(0.7, 1.3)
    args = [
        PIPER_BIN, "--model", model_path,
        "--output_file", output_path,
        "--length_scale", str(length_scale),
        "--noise_scale", str(random.uniform(0.3, 0.9)),
        "--noise_w", str(random.uniform(0.2, 0.7)),
        "--quiet",
    ]
    if speaker:
        args.extend(["--speaker", speaker])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PIPER_DIR
    result = subprocess.run(args, input=text, capture_output=True, text=True, env=env, timeout=30)
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100


def read_wav_int16(path):
    """Read 16-bit mono WAV, return int16 samples and sample rate."""
    with open(path, "rb") as f:
        data = f.read()
    sr = struct.unpack_from("<I", data, 24)[0]
    idx = data.find(b"data")
    if idx < 0:
        return None, 0
    size = struct.unpack_from("<I", data, idx + 4)[0]
    raw = data[idx + 8: idx + 8 + size]
    samples = np.frombuffer(raw, dtype=np.int16).copy()
    return samples, sr


def resample_int16(samples, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return samples
    ratio = target_sr / orig_sr
    target_len = int(len(samples) * ratio)
    if target_len < 1:
        return np.array([], dtype=np.int16)
    indices = np.linspace(0, len(samples) - 1, target_len)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(samples) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    resampled = samples[idx_floor].astype(np.float32) * (1 - frac) + samples[idx_ceil].astype(np.float32) * frac
    return resampled.astype(np.int16)


def augment_int16(samples):
    aug = samples.astype(np.float32)
    aug *= random.uniform(0.4, 1.6)
    noise_level = random.uniform(0, 600)
    aug += np.random.randn(len(aug)).astype(np.float32) * noise_level
    pad_start = int(random.uniform(0, 0.4) * 16000)
    pad_end = int(random.uniform(0, 0.3) * 16000)
    if pad_start > 0:
        aug = np.concatenate([np.zeros(pad_start, dtype=np.float32), aug])
    if pad_end > 0:
        aug = np.concatenate([aug, np.zeros(pad_end, dtype=np.float32)])
    return np.clip(aug, -32768, 32767).astype(np.int16)


def load_and_prep_wavs(wav_dir, num_augments, target_len_samples):
    """Load WAVs, resample to 16kHz, augment, pad/trim to target length.
    Returns array of shape (N, target_len_samples) as int16."""
    clips = []
    for wav_file in sorted(Path(wav_dir).glob("*.wav")):
        samples, sr = read_wav_int16(str(wav_file))
        if samples is None or len(samples) < 100:
            continue
        samples = resample_int16(samples, sr)
        for i in range(num_augments):
            if i == 0:
                clip = samples.copy()
            else:
                clip = augment_int16(samples)
            if len(clip) < target_len_samples:
                clip = np.pad(clip, (0, target_len_samples - len(clip)))
            else:
                clip = clip[:target_len_samples]
            clips.append(clip)
    return np.array(clips, dtype=np.int16) if clips else np.zeros((0, target_len_samples), dtype=np.int16)


def main():
    os.makedirs(f"{OUTPUT_DIR}/positive", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/negative", exist_ok=True)

    # Step 1: Generate WAVs
    print("=== Generating positive samples ===")
    pos_count = 0
    for phrase in POSITIVE_PHRASES:
        for model_path, speaker in VOICES:
            for _ in range(3):
                path = f"{OUTPUT_DIR}/positive/pos_{pos_count:04d}.wav"
                if generate_wav(phrase, model_path, speaker, path):
                    pos_count += 1
    print(f"Generated {pos_count} positive WAV files")

    print("\n=== Generating negative samples ===")
    neg_count = 0
    for phrase in NEGATIVE_PHRASES:
        voices = random.sample(VOICES, min(5, len(VOICES)))
        for model_path, speaker in voices:
            path = f"{OUTPUT_DIR}/negative/neg_{neg_count:04d}.wav"
            if generate_wav(phrase, model_path, speaker, path):
                neg_count += 1
    print(f"Generated {neg_count} negative WAV files")

    # Step 2: Load, augment, extract embeddings
    print("\n=== Extracting embeddings ===")
    from openwakeword.utils import AudioFeatures
    preprocessor = AudioFeatures()

    # "hey cedric" is ~1-1.5 seconds. Use 2 seconds of audio to be safe.
    target_len = 2 * 16000  # 2 seconds at 16kHz

    pos_clips = load_and_prep_wavs(f"{OUTPUT_DIR}/positive", num_augments=8, target_len_samples=target_len)
    neg_clips = load_and_prep_wavs(f"{OUTPUT_DIR}/negative", num_augments=4, target_len_samples=target_len)

    # Add noise clips
    noise_clips = []
    for _ in range(500):
        noise = (np.random.randn(target_len) * random.uniform(100, 3000)).astype(np.int16)
        noise_clips.append(noise)
    # Add silence clips
    for _ in range(200):
        noise_clips.append(np.zeros(target_len, dtype=np.int16))
    noise_clips = np.array(noise_clips, dtype=np.int16)

    print(f"Positive clips: {len(pos_clips)}")
    print(f"Negative clips: {len(neg_clips)}")
    print(f"Noise clips: {len(noise_clips)}")

    # Extract embeddings using embed_clips (batch processing)
    print("Computing positive embeddings...")
    pos_emb = preprocessor.embed_clips(pos_clips, batch_size=64)
    print(f"  Shape: {pos_emb.shape}")  # (N, frames, 96)

    print("Computing negative embeddings...")
    neg_emb = preprocessor.embed_clips(neg_clips, batch_size=64)
    print(f"  Shape: {neg_emb.shape}")

    print("Computing noise embeddings...")
    noise_emb = preprocessor.embed_clips(noise_clips, batch_size=64)
    print(f"  Shape: {noise_emb.shape}")

    # Step 3: Create training windows of INPUT_WINDOWS from embeddings
    def extract_windows(embeddings, window_size):
        """From (N, T, 96) extract all windows of (window_size, 96)."""
        windows = []
        for i in range(len(embeddings)):
            T = embeddings.shape[1]
            if T < window_size:
                # Pad
                padded = np.zeros((window_size, WW_FEATURES), dtype=np.float32)
                padded[-T:] = embeddings[i]
                windows.append(padded)
            else:
                # Slide windows
                for start in range(T - window_size + 1):
                    windows.append(embeddings[i, start:start + window_size])
        return windows

    pos_windows = extract_windows(pos_emb, INPUT_WINDOWS)
    neg_windows = extract_windows(neg_emb, INPUT_WINDOWS)
    noise_windows = extract_windows(noise_emb, INPUT_WINDOWS)

    neg_windows.extend(noise_windows)

    print(f"\nPositive windows: {len(pos_windows)}")
    print(f"Negative windows: {len(neg_windows)}")

    if len(pos_windows) < 10:
        print("ERROR: Too few positive samples")
        return

    X_pos = np.array(pos_windows, dtype=np.float32)
    X_neg = np.array(neg_windows, dtype=np.float32)

    # Balance: at most 3x negatives
    if len(X_neg) > len(X_pos) * 3:
        idx = np.random.choice(len(X_neg), len(X_pos) * 3, replace=False)
        X_neg = X_neg[idx]

    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos), dtype=np.float32),
                        np.zeros(len(X_neg), dtype=np.float32)])

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Total: {len(X)}, positive ratio: {y.mean():.2%}")
    print(f"Shape: {X.shape}")  # (N, 16, 96)

    # Step 4: Train
    print("\n=== Training model ===")
    import tensorflow as tf

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_WINDOWS, WW_FEATURES)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
        ],
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.2%}")
    print(f"Validation loss: {val_loss:.4f}")

    # Step 5: Export to TFLite
    print("\n=== Exporting to TFLite ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(FINAL_MODEL, "wb") as f:
        f.write(tflite_model)
    print(f"Model saved to {FINAL_MODEL} ({len(tflite_model) / 1024:.1f} KB)")

    # Verify
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print(f"TFLite input: {inp[0]['shape']} {inp[0]['dtype']}")
    print(f"TFLite output: {out[0]['shape']} {out[0]['dtype']}")

    # Quick test
    interpreter.set_tensor(inp[0]['index'], X_val[:1])
    interpreter.invoke()
    pred = interpreter.get_tensor(out[0]['index'])
    print(f"Test: label={y_val[0]:.0f}, pred={pred[0][0]:.4f}")


if __name__ == "__main__":
    main()
