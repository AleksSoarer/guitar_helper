import pyaudio
import numpy as np
import soundfile as sf
import keyboard

# Параметры записи
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Функция для записи аудио до нажатия клавиши
def record_audio(output_filename):
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

# Запись 10 образцов мелодий
for i in range(10):
    print(f"Recording sample {i+1}/10")
    record_audio(f'sample_{i+1}.wav')

print("All samples saved.")