import os
import pyaudio
import wave
import librosa, librosa.display
from librosa.core import spectrum
import matplotlib.pyplot as plt
import numpy as np

print("...WELCOME TO PYTHON DSP...")
rt = int(input("Enter recording time:"))
name = input("The name of record:")
print("recording time is: " + str(rt))

#create dir for analys figures
dir_data = name + '_data'
dir_sound = 'records'
try:
    os.mkdir(dir_data)
    os.mkdir(dir_sound)
    print("Directory " , dir_data ,  " Created!")
    print("Directory"  , dir_sound , "Created!") 
except FileExistsError:
    print("Directory " , dir_data ,  " already exists!")
    print("Directory"  , dir_sound , "already exist!")

#route dir
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, dir_data + '/')


chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = rt
filename = name + ".wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks 
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording. Start analysis')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()



FIG_SIZE = (3,4)

sound = name + ".wav"

#plot waveform sr = sample rate
sound, sr = librosa.load(sound, sr = 22050)

#waveform
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(sound, sr = sr, alpha=0.4)
plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.title(name + "  Waveform", fontweight='bold')
#save waveform to _data dir
sample_file_name = "waveform"
plt.savefig(results_dir + sample_file_name)

#spectogram
n_fft = 2048
hop_length = 512

#print("STFT hop length duration is: {}s".format(hop_length_duration))
#print("STFT window duration is: {}s".format(n_fft_duration))

stft = librosa.core.stft(sound, hop_length=hop_length, n_fft=n_fft)
spectogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectogram)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title(name + "  Spectrogram (dB)", fontweight='bold')
sample_file_name = "spectogram"
plt.savefig(results_dir + sample_file_name)


#spectrum
plt.figure(figsize=FIG_SIZE)
fft_spectrum = np.fft.rfft(sound)
freq = np.fft.rfftfreq(sound.size, d=1./sr)
fft_spectrum_abs = np.abs(fft_spectrum)
plt.plot(freq[:4000], fft_spectrum_abs[:4000])
plt.plot(freq, fft_spectrum_abs)
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.title(name + "  Frequency Domain", fontweight='bold')
plt.xscale("log")
plt.grid()
sample_file_name = "spectrum"
plt.savefig(results_dir + sample_file_name)


# show plots
plt.show()

