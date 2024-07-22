import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Функция для загрузки и визуализации аудио
def load_and_plot(file_path, title):
    audio, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(14, 5))
    plt.title(title)
    librosa.display.waveshow(audio, sr=sr)
    plt.show()
    return audio, sr

# Функция для обрезки аудио
def trim_audio(audio, start_time, end_time, sr):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    trimmed_audio = audio[start_sample:end_sample]
    return trimmed_audio

# Функция для сохранения аудио
def save_audio(audio, sr, output_path):
    sf.write(output_path, audio, sr)

# Путь к исходному аудио файлу
input_file = 'sample_7.wav'  # Укажите путь к вашему файлу

# Загрузка и визуализация исходного аудио
audio, sr = load_and_plot(input_file, 'Original Audio')

# Определение начала и конца интересующего фрагмента в секундах
start_time = 1.3  # Начало интересующего фрагмента (в секундах)
end_time = len(audio) / sr - 0.3  # Конец интересующего фрагмента (в секундах)

# Обрезка аудио
trimmed_audio = trim_audio(audio, start_time, end_time, sr)

# Визуализация обрезанного аудио
plt.figure(figsize=(14, 5))
plt.title('Trimmed Audio')
librosa.display.waveshow(trimmed_audio, sr=sr)
plt.show()

# Сохранение обрезанного аудио
output_file = 'trimmed_sample_7.wav'  # Укажите путь для сохранения файла
save_audio(trimmed_audio, sr, output_file)