import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
from python_speech_features import mfcc
from scipy.signal.windows import hamming
from matplotlib.font_manager import FontProperties
from keras.models import load_model as keras_load_model
from tkinter import messagebox

# 使用黑体字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def load_saved_model(model_path):
    global model
    model = keras_load_model(model_path)

def load_wav_file():
    global wav_file
    wav_file = filedialog.askopenfilename(filetypes=[('WAV files', '*.wav')])
    rate, data = wavfile.read(wav_file)
    plot_waveform(rate, data)

def plot_waveform(rate, data):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title('待测语音原始波形')
    ax.set_xlabel('时间')
    ax.set_ylabel('振幅')
    waveform_plot = FigureCanvasTkAgg(fig, window)
    waveform_plot.get_tk_widget().grid(row=2, column=0)

def extract_features():
    global wav_file, extracted_features
    rate, data = wavfile.read(wav_file)
    extracted_features = extract_features_from_audio(data, rate)

def extract_features_from_audio(audio_data, audio_fs):
    # 预处理参数
    target_sample_rate = 16000  # 重采样目标采样率（Hz）
    frame_length = 0.025  # 帧长度（s）
    frame_overlap = 0.01  # 帧重叠（s）

    # 重采样
    if audio_fs != target_sample_rate:
        audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=audio_fs,
                                      target_sr=target_sample_rate)
        audio_fs = target_sample_rate

    # 转换为单声道
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 提取音色特征 (MFCC)
    mfccs = mfcc(audio_data, audio_fs)

    # 提取音高特征
    pitch_values = librosa.yin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_values = np.mean(pitch_values) * np.ones((mfccs.shape[0], 1))

    # 计算音强特征
    frame_length_samples = int(round(frame_length * audio_fs))
    frame_overlap_samples = int(round(frame_overlap * audio_fs))
    rms_window = hamming(frame_length_samples, sym=False)
    frame_starts = np.arange(0, len(audio_data) - frame_length_samples + 1, frame_overlap_samples)
    rms_values = np.zeros((len(frame_starts), 1))
    for i in range(len(frame_starts)):
        frame = audio_data[frame_starts[i]:frame_starts[i] + frame_length_samples]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))
    rms_values = rms_values[:mfccs.shape[0], :]

    # 将所有特征裁剪为相同的长度
    min_len = min(mfccs.shape[0], rms_values.shape[0], pitch_values.shape[0])
    mfccs = mfccs[:min_len, :]
    rms_values = rms_values[:min_len, :]
    pitch_values = pitch_values[:min_len, :]

    # 提取持续时间特征
    duration_value = len(audio_data) / audio_fs

    # 将特征堆叠在一起
    stacked_features = np.hstack((pitch_values, mfccs, rms_values, np.full((mfccs.shape[0], 1), duration_value)))

    return stacked_features

def plot_features():
    global extracted_features

    # 获取各种特征的长度
    pitch_length = len(extracted_features[:, 0])
    mfcc_length = extracted_features.shape[1] - 2 - 1
    rms_length = len(extracted_features[:, -2])
    duration_length = len(extracted_features[:, -1])

    # 创建一个带有4个子图的画布
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # 绘制音高特征
    axes[0].plot(np.arange(pitch_length), extracted_features[:, 0], color='blue', label='Pitch')
    axes[0].set_title('音高特征')
    axes[0].set_xlabel('时间')
    axes[0].set_ylabel('频率')
    axes[0].legend()

    # 绘制音色特征 (MFCC)
    img = axes[1].imshow(extracted_features[:, 1:1+mfcc_length].T, origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_title('音色特征')
    axes[1].set_xlabel('时间')
    axes[1].set_ylabel('MFCC系数')
    fig.colorbar(img, ax=axes[1], label='MFCC Value')

    # 绘制音强特征
    axes[2].plot(np.arange(rms_length), extracted_features[:, -2], color='green', label='RMS')
    axes[2].set_title('音强特征')
    axes[2].set_xlabel('时间')
    axes[2].set_ylabel('均方根振幅')
    axes[2].legend()

    # 绘制持续时间特征
    axes[3].bar(np.arange(duration_length), extracted_features[:, -1], color='red', label='Duration')
    axes[3].set_title('持续时间特征')
    axes[3].set_xlabel('时间')
    axes[3].set_ylabel('持续时间')
    axes[3].legend()

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()

def detect_emotion():
    global model, extracted_features

    # 预处理特征以匹配模型输入
    input_features = np.expand_dims(extracted_features, axis=0)  # 增加批次维度

    # 调整输入特征的形状
    n_features = extracted_features.shape[1]
    input_features = input_features.reshape(-1, n_features, 1)

    # 预测情感类别概率
    emotion_probabilities = model.predict(input_features)[0]

    # 获取最大概率对应的情感标签
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    predicted_emotion = emotions[np.argmax(emotion_probabilities)]

    # 显示预测结果
    tk.messagebox.showinfo('预测结果', f'预测情感: {predicted_emotion}\n各类别概率: {emotion_probabilities}')

# 创建主窗口
window = tk.Tk()
window.title('语音情感识别')

# 创建并放置按钮
load_model_button = tk.Button(window, text='加载模型', command=lambda: load_saved_model(filedialog.askopenfilename(filetypes=[('HDF5 files', '*.h5')])))
load_model_button.grid(row=0, column=0)

load_wav_button = tk.Button(window, text='加载WAV文件', command=load_wav_file)
load_wav_button.grid(row=1, column=0)

extract_features_button = tk.Button(window, text='提取特征', command=extract_features)
extract_features_button.grid(row=3, column=0)

plot_features_button = tk.Button(window, text='显示特征', command=plot_features)
plot_features_button.grid(row=4, column=0)

detect_emotion_button = tk.Button(window, text='检测情感', command=detect_emotion)
detect_emotion_button.grid(row=5, column=0)

# 运行主循环
window.mainloop()
