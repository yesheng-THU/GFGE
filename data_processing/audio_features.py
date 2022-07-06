import numpy as np
import librosa
import math
import os
import scipy.io.wavfile as wav

def extract_melspec1(X, fps, fs):
    mel_all = {}


    X = X.astype(float) / math.pow(2, 15)

    # assert fs % fps == 0

    hop_len = int(round(fs / fps))

    n_fft = int(fs * 0.13)
    C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=n_fft, hop_length=hop_len, n_mels=29, fmin=0.0, fmax=8000)
    C = np.log(C + 1e-6)
    print("fs: " + str(fs))
    print(C.shape)
    print(np.min(C), np.max(C))
    C_use = np.transpose(C)
    C_out = []
    for i in range(C_use.shape[0]):
        if np.sum(C_use[i]) != float("-inf") and np.sum(C_use[i]) != float("inf"):
            C_out.append(C_use[i])
    C_out = np.array(C_out)
    return C_out

def extract_melspec(audio_dir, files, destpath, fps):

    for f in files:
        file = os.path.join(audio_dir, f + '.wav')
        outfile = destpath + '/' + f + '.npy'
        
        print('{}\t->\t{}'.format(file,outfile))
        fs,X = wav.read(file)
        X = X.astype(float)/math.pow(2,15)
        
        assert fs%fps == 0
        
        hop_len=int(fs/fps)
        
        n_fft=int(fs*0.13)
        C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=2048, hop_length=hop_len, n_mels=27, fmin=0.0, fmax=8000)
        C = np.log(C)
        print("fs: " + str(fs))
        print("hop_len: " + str(hop_len))
        print("n_fft: " + str(n_fft))
        print(C.shape)
        print(np.min(C),np.max(C))
        np.save(outfile,np.transpose(C))