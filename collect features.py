import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf

# Функция для извлечения MFCC признаков и приведения к одинаковой длине
def extract_features(file_name, max_pad_len=100):
    audio, sample_rate = librosa.load(file_name, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs, audio, sample_rate

# Функция для аугментации данных
def augment_data(audio, sample_rate):
    augmented = []
    augmented.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=4))  # Pitch shifting up
    augmented.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-4)) # Pitch shifting down
    augmented.append(librosa.effects.time_stretch(audio, rate=0.8))  # Time stretching
    augmented.append(librosa.effects.time_stretch(audio, rate=1.2))  # Time shrinking
    augmented.append(audio + 0.005 * np.random.randn(len(audio)))  # Добавление шума
    return augmented

# Извлечение признаков и аугментация данных для всех образцов
features = []
labels = []

# Примеры меток для данных с разным количеством повторений
repetitions = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]  # Замените на реальные метки

for i in range(10):
    file_name = f'trimmed_sample_{i+1}.wav'
    mfccs, audio, sr = extract_features(file_name)
    features.append(mfccs)
    labels.append(repetitions[i])
    
    augmented_audios = augment_data(audio, sr)
    for augmented_audio in augmented_audios:
        augmented_mfccs = librosa.feature.mfcc(y=augmented_audio, sr=sr, n_mfcc=40)
        pad_width = 100 - augmented_mfccs.shape[1]
        if pad_width > 0:
            augmented_mfccs = np.pad(augmented_mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            augmented_mfccs = augmented_mfccs[:, :100]
        features.append(augmented_mfccs)
        labels.append(repetitions[i])

features = np.array(features)
labels = np.array(labels)

np.save('features_augmented.npy', features)
np.save('labels_augmented.npy', labels)