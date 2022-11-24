import numpy as np
import os
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import zscore


#model 
model_path = './models'
model_name = 'SER_2DCNN_LSTM_eng.h5'
model = load_model(os.path.join(model_path, model_name))

# constants
starttime = datetime.now()

#emotion_labels
labels = [0,1,2,3,4,5,6]
emotions = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']
emotion_lables = dict(zip(labels, emotions))

CAT7 = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {"positive": "green",
              "negative": "red",
              "neutral": "grey",
              "angry":   "red",
              "disgust": "brown",
              "fearful": "purple",
              "happy": "orange",
              "surprise": "yellow",
              "sad": "blue",
              }

TEST_CAT = ['angry', 'disgust', 'fearul', 'happy', 'neutral', 'sad', 'surprise']
TEST_PRED = np.array([.1, .3, .3, .1, .4, .6, .9])
backgroundColor = "#6c6c71"

def read_audio(wav_file):
    max_pad_len = 49100
    sig, sr = librosa.load(wav_file, sr=16000, res_type='kaiser_fast', duration=3, offset=0.5)
    sr = np.array(sr)
    y = zscore(sig)
        
    # Padding or truncated signal 
    if len(y) < max_pad_len:    
        y_padded = np.zeros(max_pad_len)
        y_padded[:len(y)] = y
        y = y_padded
    elif len(y) > max_pad_len:
        y = np.asarray(y[:max_pad_len])
    return y, sr

def get_melspec(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    
    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    
    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    return mel_spect

def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames

def get_title(predictions, categories, first_line=''):
    txt = f"{first_line}\nDetected emotion: \
  {categories[predictions.argmax()]} - {predictions.max() * 100:.2f}%"
    return txt


def plot_colored_polar(fig, predictions, categories,
                        title="", colors=COLOR_DICT):
    N = len(predictions)
    ind = predictions.argmax()

    COLOR = sector_color = colors[categories[ind]]
    sector_colors = [colors[i] for i in categories]

    fig.set_facecolor(backgroundColor)
    ax = plt.subplot(111, polar="True")

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    for sector in range(predictions.shape[0]):
        radii = np.zeros_like(predictions)
        radii[sector] = predictions[sector] * 10
        width = np.pi / 1.8 * predictions
        c = sector_colors[sector]
        ax.bar(theta, radii, width=width, bottom=0.0, color=c, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)

    plt.suptitle(title, color="white", size=10)
    plt.title(f"Based on {N} emotion categoreis {N}\n", color='white')
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)

def plot_melspec(path, model=None, three=False,
                 CAT3=CAT3, CAT7=CAT7):
    # load model if it is not loaded
    if model is None:
        model = load_model(os.path.join(model_path, model_name))
    # mel-spec model results
    wav, sr = read_audio(path)
    mfcc = get_melspec(wav)
    mfcc = mfcc.reshape(1, *mfcc.shape)
    win_ts = 128
    hop_ts = 64
    # Frame for TimeDistributed model
    X = frame(mfcc, hop_ts, win_ts)
    X_= X.reshape(*X.shape, 1)
    pred = model.predict(X_)[0, :]
    cat = CAT7

    if three:
        pos = pred[3] + pred[6] * .5  # happy and surprised
        neu = pred[4] + pred[5] * .5 + pred[6] * .5  # neutral, surprised, and sadness
        neg = pred[0] + pred[1] + pred[2] + pred[4] * .5  # fear, disgust, anger and sadness
        pred = np.array([pos, neu, neg])
        pred = np.array([pos, neu, neg])
        cat = CAT3

    txt = get_title(pred, cat)
    fig = plt.figure(figsize=(6, 4))
    plot_colored_polar(fig, predictions=pred, categories=cat, title=txt)
    return (fig, pred)

if __name__ == "__main__":
    plot_melspec("test.wav")