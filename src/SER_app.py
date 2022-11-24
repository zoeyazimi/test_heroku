import os
import pandas
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import zscore
import streamlit as st
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components
from melspec import plot_colored_polar

# load models
# model = load_model("model3.h5")
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
              "happy": "green",
              "surprise": "yellow",
              "sad": "blue",
              }

TEST_CAT = ['angry', 'disgust', 'fearul', 'happy', 'neutral', 'sad', 'surprise']
TEST_PRED = np.array([.1, .3, .3, .1, .4, .6, .9])


# page settings
st.set_page_config(page_title="SER web-app", page_icon=":shark:", layout="wide")
backgroundColor = "#6c6c71"



# @st.cache
def log_file(txt=None):
    with open("log.txt", "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")


def save_audio(file):
    if file.size > 4000000:
        return 1
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


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

# This part is used for gender detection
def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

def get_mel_spect(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    
    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    
    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    return mel_spect

# Split spectrogram into frames (for LSTM)
def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames


@st.cache
def get_title(predictions, categories=CAT7):
    title = f"Predicted emotion is {categories[predictions.argmax()]} \
    with probabilty of {predictions.max() * 100:.2f}%"
    return title

def main():
    side_img = Image.open("images/SER.jpg")
    with st.sidebar:
        st.image(side_img, width=300)
    st.sidebar.subheader("Functions")
    website_menu = st.sidebar.selectbox("Menu", ("Emotion Recognition", "Project description", "Team"))
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if website_menu == "Emotion Recognition":
        st.sidebar.subheader("Model")
        model_type = st.sidebar.selectbox("Choose the feature for prediction", ("mfccs", "opensmile"))
        em3=em7=gender = False
        st.sidebar.subheader("Settings")
        st.markdown("## Upload the file")
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
                if audio_file is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audio_file.name)
                    if_save_audio = save_audio(audio_file)
                    if if_save_audio == 1:
                        st.warning("File size is too large. Try another file.")
                    elif if_save_audio == 0:

                        st.audio(audio_file, format='audio/wav', start_time=0)
                        try:
                            wav, sr = read_audio(path)
                            Xdb = get_mel_spect(wav)
                            mfccs = librosa.feature.mfcc(wav, sr=sr)
                        except Exception as e:
                            audio_file = None
                            st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
                    else:
                        st.error("Unknown error")
                else:
                    if st.button("Try test file"):
                        wav, sr = read_audio("test.wav")
                        Xdb = get_mel_spect(wav)
                        mfccs = librosa.feature.mfcc(wav, sr=sr)
                        # display audio
                        st.audio("test.wav", format='audio/wav', start_time=0)
                        path = "test.wav"
                        audio_file = "test"
            with col2:
                if audio_file is not None:
                    fig = plt.figure(figsize=(10, 2))
                    fig.set_facecolor(backgroundColor)
                    plt.title("Wave-form")
                    librosa.display.waveplot(wav, sr=sr)
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    plt.gca().axes.spines["bottom"].set_visible(False)
                    plt.gca().axes.set_facecolor(backgroundColor)
                    st.write(fig)
                else:
                    pass

        if model_type == "mfccs":
            em3 = st.sidebar.checkbox("3 categories of emotions", True)
            em7 = st.sidebar.checkbox("7 categories of emotions", True)
            gender = st.sidebar.checkbox("gender recognition")

        elif model_type == "mel-specs":
            st.sidebar.warning("This model is temporarily disabled")

        else:
            st.sidebar.warning("This model is temporarily disabled")


        if audio_file is not None:
            st.markdown("## Analyzing...")
            if not audio_file == "test":
                st.sidebar.subheader("Audio file")
                file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
                st.sidebar.write(file_details)

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    fig = plt.figure(figsize=(10, 2))
                    fig.set_facecolor(backgroundColor)
                    plt.title("MFCCs")
                    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig)
                with col2:
                    fig2 = plt.figure(figsize=(10, 2))
                    fig2.set_facecolor(backgroundColor)
                    plt.title("Mel-log-spectrogram")
                    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig2)

            if model_type == "mfccs":
                st.markdown("## Predictions")
                with st.container():
                    col1, col2= st.columns(2)
                    wav, sr = read_audio(path)
                    mfcc = get_mel_spect(wav)
                    mfcc = mfcc.reshape(1, *mfcc.shape)
                    win_ts = 128
                    hop_ts = 64
                    # Frame for TimeDistributed model
                    X = frame(mfcc, hop_ts, win_ts)
                    X_= X.reshape(X.shape[0], X.shape[1] , X.shape[2], X.shape[3], 1)
                    pred = model.predict(X_)[0,:]
                    # with col1:
                    #     if em3:
                    #         pos = pred[3] + pred[6] * .5  # happy and surprised
                    #         neu = pred[4] + pred[5] * .5 + pred[6] * .5  # neutral, surprised, and sadness
                    #         neg = pred[0] + pred[1] + pred[2] + pred[4] * .5  # fear, disgust, anger and sadness
                    #         data3 = np.array([pos, neu, neg])
                    #         txt = get_title(data3, CAT3)
                    #         fig = plt.figure(figsize=(5, 5))
                    #         COLORS =COLOR_DICT
                    #         plot_colored_polar(fig, predictions=data3, categories=CAT3,
                    #                            title=txt, colors=COLORS)

                    #         st.write(fig)
                    with col1:
                        if em7:
                            mfcc = get_mel_spect(wav)
                            # mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
                            mfcc = mfcc.reshape(1, *mfcc.shape)
                            win_ts = 128
                            hop_ts = 64
                            # Frame for TimeDistributed model
                            X = frame(mfcc, hop_ts, win_ts)
                            X_= X.reshape(X.shape[0], X.shape[1] , X.shape[2], X.shape[3], 1)
                            pred = model.predict(X_)[0,:]
                            txt = get_title(pred, CAT7)
                            fig1 = plt.figure(figsize=(5, 5))
                            COLORS = COLOR_DICT
                            plot_colored_polar(fig1, predictions=pred, categories=CAT7,
                                               title=txt, colors=COLORS)

                            st.write(fig1)
                    with col2:
                        if gender:
                            with st.spinner('Wait for it...'):
                                gmodel = load_model("./models/model_mw.h5")
                                gmfccs = get_mfccs(path, gmodel.input_shape[-1])
                                gmfccs = gmfccs.reshape(1, *gmfccs.shape)
                                gpred = gmodel.predict(gmfccs)[0]
                                gdict = [["female", "female.png"], ["male", "male.png"]]
                                ind = gpred.argmax()
                                txt = "Predicted gender: " + gdict[ind][0]
                                img = Image.open("images/" + gdict[ind][1])

                                fig2 = plt.figure(figsize=(3, 3))
                                fig2.set_facecolor(backgroundColor)
                                plt.title(txt, color="white")
                                plt.imshow(img)
                                plt.axis("off")
                                st.write(fig2)

    elif website_menu == "Project description":
        import pandas as pd
        import plotly.express as px
        st.title("Project description")
        st.subheader("GitHub")
        link = '[GitHub repository of the web-application (placeholder)]' \
               '(https://github.com/IXP-Team)'
        st.markdown(link, unsafe_allow_html=True)

        st.subheader("Theory")
        link = '[This work is based on: - link the paper]' \
               '(https://github.com/IXP-Team)'
        st.markdown(link + ":clap::clap::clap: Tal!", unsafe_allow_html=True)
        with st.expander("See Wikipedia definition"):
            components.iframe("https://en.wikipedia.org/wiki/Emotion_recognition",
                              height=320, scrolling=True)

        st.subheader("Dataset")
        txt = """
            This web-application is ... . 

            Datasets used in this project
            * Crowd-sourced Emotional Mutimodal Actors Dataset (**Crema-D**)
            * Ryerson Audio-Visual Database of Emotional Speech and Song (**Ravdess**)
            * Surrey Audio-Visual Expressed Emotion (**Savee**)
            * Toronto emotional speech set (**Tess**)    
            """
        st.markdown(txt, unsafe_allow_html=True)

        ## Add EDA here  
        # df = pd.read_csv("meta_data.csv")
        # fig = px.violin(df, y="source", x="emotions", color="actors", box=True, points="all", hover_data=df.columns)
        # st.plotly_chart(fig, use_container_width=True)


    elif website_menu == "Team":
        st.subheader("Team")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.info("z.azimi@ixp-duesseldorf.de")
        with col2:
            liimg = Image.open("images/LI-Logo.png")
            st.image(liimg)
            st.markdown(f""":speech_balloon: [Zohre Azimi](www.linkedin.com/in/zohre-azimi)""",
                        unsafe_allow_html=True)

if __name__ == '__main__':
    main()
