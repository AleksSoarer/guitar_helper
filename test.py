import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import pyaudio
import keyboard

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

# Функция для записи аудио до нажатия клавиши
def record_audio(output_filename):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print(f"Recording... Press 'q' to stop.")
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))
        if keyboard.is_pressed('q'):
            print("Recording stopped.")
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_data = np.hstack(frames)
    sf.write(output_filename, audio_data, RATE)
    print(f"Audio recorded and saved to {output_filename}")
    
    # Преобразование аудио данных в формат с плавающей точкой для визуализации
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
    
    # Визуализация записанного аудио
    plt.figure(figsize=(14, 5))
    plt.title("Recorded Audio")
    librosa.display.waveshow(audio_data, sr=RATE)
    plt.show()

# Загрузка модели
model = tf.keras.models.load_model('melody_repetition_model_cnn_lstm_regularized.keras')

# Запись новой мелодии
print("Recording test melody.")
record_audio('test_melody.wav')

# Визуализация первого примера из обучающего набора
features = np.load('features_augmented.npy')
mfccs, audio, sr = extract_features('trimmed_sample_1.wav')

plt.figure(figsize=(14, 5))
plt.title("First Training Example Audio")
librosa.display.waveshow(audio, sr=sr)
plt.show()

# Визуализация тестовой записи
mfccs, test_audio, test_sr = extract_features('test_melody.wav')

plt.figure(figsize=(14, 5))
plt.title("Test Melody Audio")
librosa.display.waveshow(test_audio, sr=test_sr)
plt.show()

# Функция для предсказания повторений
def predict_repetitions(model, file_name, max_pad_len=100):
    mfccs, audio, sample_rate = extract_features(file_name, max_pad_len)
    mfccs = np.expand_dims(mfccs.T, axis=(0, -1))
    prediction = model.predict(mfccs)
    return prediction

# Предсказание
prediction = predict_repetitions(model, 'test_melody.wav')

print(f"Predicted repetitions: {int(round(prediction[0][0]))}")