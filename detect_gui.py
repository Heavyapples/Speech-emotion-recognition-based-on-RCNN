import sys
import os
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
from python_speech_features import mfcc
from scipy.signal.windows import hamming
from keras.models import load_model
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EmotionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.wav_file = None
        self.extracted_features = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('语音情感识别')
        self.resize(800, 600)

        # Buttons
        self.btn_load_model = QtWidgets.QPushButton('加载模型')
        self.btn_load_wav = QtWidgets.QPushButton('加载WAV文件')
        self.btn_extract = QtWidgets.QPushButton('提取特征')
        self.btn_plot = QtWidgets.QPushButton('显示特征')
        self.btn_detect = QtWidgets.QPushButton('检测情感')

        # Canvas for waveform
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Layouts
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.btn_load_model)
        btn_layout.addWidget(self.btn_load_wav)
        btn_layout.addWidget(self.btn_extract)
        btn_layout.addWidget(self.btn_plot)
        btn_layout.addWidget(self.btn_detect)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)

        # Connections
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_wav.clicked.connect(self.load_wav)
        self.btn_extract.clicked.connect(self.extract_features)
        self.btn_plot.clicked.connect(self.plot_features)
        self.btn_detect.clicked.connect(self.detect_emotion)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择模型文件', filter='HDF5 files (*.h5)')
        if path:
            self.model = load_model(path)

    def load_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择WAV文件', filter='WAV files (*.wav)')
        if path:
            self.wav_file = path
            rate, data = wavfile.read(self.wav_file)
            self.plot_waveform(rate, data)

    def plot_waveform(self, rate, data):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data)
        ax.set_title('待测语音原始波形', fontproperties=font)
        ax.set_xlabel('时间', fontproperties=font)
        ax.set_ylabel('振幅', fontproperties=font)
        self.canvas.draw()

    def extract_features(self):
        if not self.wav_file:
            return
        rate, data = wavfile.read(self.wav_file)
        self.extracted_features = self._extract_from_audio(data, rate)

    def _extract_from_audio(self, audio_data, audio_fs):
        target_sr = 16000
        frame_length = 0.025
        frame_overlap = 0.01

        if audio_fs != target_sr:
            audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=audio_fs, target_sr=target_sr)
            audio_fs = target_sr
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        mfccs = mfcc(audio_data, audio_fs)
        pitch = librosa.yin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch = np.mean(pitch) * np.ones((mfccs.shape[0], 1))

        fls = int(round(frame_length * audio_fs))
        fos = int(round(frame_overlap * audio_fs))
        starts = np.arange(0, len(audio_data) - fls + 1, fos)
        rms = np.array([np.sqrt(np.mean(audio_data[s:s+fls]**2)) for s in starts])
        rms = rms[:mfccs.shape[0]].reshape(-1,1)

        min_len = min(mfccs.shape[0], rms.shape[0], pitch.shape[0])
        mfccs = mfccs[:min_len]
        rms = rms[:min_len]
        pitch = pitch[:min_len]

        duration = len(audio_data) / audio_fs
        dur_arr = np.full((min_len,1), duration)

        return np.hstack((pitch, mfccs, rms, dur_arr))

    def plot_features(self):
        if self.extracted_features is None:
            return
        # 弹出新窗口展示特征
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('特征展示')
        dlg.resize(900, 700)

        # 创建带更大尺寸的 Figure
        fig = plt.figure(figsize=(9, 7))
        # 调整子图间距
        fig.subplots_adjust(hspace=0.5)
        canvas = FigureCanvas(fig)

        feat = self.extracted_features
        gs = fig.add_gridspec(4, 1)

        # 音高
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(feat[:, 0])
        ax0.set_title('音高特征', fontproperties=font)
        ax0.set_ylabel('频率(Hz)', fontproperties=font)

        # MFCC
        ax1 = fig.add_subplot(gs[1, 0])
        img = ax1.imshow(feat[:, 1:-2].T, origin='lower', aspect='auto')
        ax1.set_title('音色特征 (MFCC)', fontproperties=font)
        ax1.set_ylabel('系数', fontproperties=font)
        fig.colorbar(img, ax=ax1, fraction=0.02, pad=0.04)

        # RMS
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.plot(feat[:, -2])
        ax2.set_title('音强特征 (RMS)', fontproperties=font)
        ax2.set_ylabel('幅度', fontproperties=font)

        # 持续时间
        ax3 = fig.add_subplot(gs[3, 0])
        ax3.bar(np.arange(len(feat[:, -1])), feat[:, -1])
        ax3.set_title('持续时间特征', fontproperties=font)
        ax3.set_ylabel('时长(s)', fontproperties=font)

        btn_save = QtWidgets.QPushButton('下载图片')
        btn_save.clicked.connect(lambda: self.save_figure(fig))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(btn_save)
        dlg.setLayout(layout)

        canvas.draw()
        dlg.exec_()

    def save_figure(self, fig):
        path, _ = QFileDialog.getSaveFileName(self, '保存图片', filter='PNG files (*.png)')
        if path:
            fig.savefig(path)

    def detect_emotion(self):
        if self.model is None or self.extracted_features is None:
            return
        inp = self.extracted_features[np.newaxis, ...]
        inp = inp.reshape(-1, inp.shape[2], 1)
        probs = self.model.predict(inp)[0]
        labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        idx = np.argmax(probs)
        text = f"预测情感: {labels[idx]}\n"
        text += '\n'.join([f"{l}: {p*100:.2f}%" for l, p in zip(labels, probs)])

        dlg = QMessageBox(self)
        dlg.setWindowTitle('预测结果')
        dlg.setText(text)
        dlg.exec_()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = EmotionApp()
    win.show()
    sys.exit(app.exec_())