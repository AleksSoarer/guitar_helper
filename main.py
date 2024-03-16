import pyaudio
import numpy as np
import librosa
import wave

# Функция для записи аудио с микрофона
def record_audio(seconds, rate=44100, chunk_size=1024):
    file_path = 'C:\\Test\\Guitar helper\\test.wav'
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    frames = []
    for _ in range(0, int(rate / chunk_size * seconds)):
        data = stream.read(chunk_size)
        frames.append(np.frombuffer(data, dtype=np.int16))
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Сохраняем аудио в файл WAV
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return np.concatenate(frames).astype(np.float32)  # Исправлено: приводим к типу float32

# Функция для распознавания начала и конца мелодии
def detect_melody(audio_data, threshold=0.1):
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=44100)
    start = onset_frames[0] if len(onset_frames) > 0 else 0
    end = onset_frames[-1] if len(onset_frames) > 0 else len(audio_data)
    print("start = ", start, 'end = ', end , '\n')
    return start, end

# Главная функция
def main():
    print("Запись...")
    # Записываем аудио в течение 10 секунд
    audio_data = record_audio(seconds=10)
    print('rec ok!')
    # Распознаем начало и конец мелодии
    start, end = detect_melody(audio_data)
    
    # Определяем, сколько раз мелодия была сыграна
    melody_count = 0
    while True:
        print("Запись...")
        audio_data = record_audio(seconds=10)  # Записываем аудио в течение 10 секунд
        
        new_start, new_end = detect_melody(audio_data)
        if new_start < start or new_end > end:
            melody_count += 1
            start = new_start
            end = new_end
            print("Melody played! Count:", melody_count)

if __name__ == "__main__":
    main()
