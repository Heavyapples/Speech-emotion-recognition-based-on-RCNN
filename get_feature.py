import os
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
from python_speech_features import mfcc
from scipy.signal.windows import hamming

# 设置文件夹路径
root_path = r'E:\代码接单\rcnn语音情感识别\project2_database\enterface database'
subject_folders = [folder for folder in os.listdir(root_path) if folder.startswith('subject')]
emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# 预处理参数
target_sample_rate = 16000  # 重采样目标采样率（Hz）
frame_length = 0.025  # 帧长度（s）
frame_overlap = 0.01  # 帧重叠（s）

# 初始化特征矩阵和标签向量
pitch_features = None
timbre_features = None
loudness_features = None
duration_features = None
labels = None

# 遍历所有志愿者文件夹
for subject_index in range(len(subject_folders)):
    subject_folder = os.path.join(root_path, subject_folders[subject_index])

    # 遍历所有情感文件夹
    for emotion_index in range(len(emotions)):
        emotion_folder = os.path.join(subject_folder, emotions[emotion_index])

        # 遍历所有句子文件夹
        for sentence_index in range(1, 6):
            sentence_folder = os.path.join(emotion_folder, f'sentence {sentence_index}')

            # 检查句子文件夹是否存在
            if os.path.isdir(sentence_folder):
                wav_files = [file for file in os.listdir(sentence_folder) if file.endswith('.wav')]

                # 检查WAV文件是否存在
                if wav_files:
                    wav_file_path = os.path.join(sentence_folder, wav_files[0])

                    # 读取音频文件
                    audio_fs, audio_data = wavfile.read(wav_file_path)

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

                    # 提取持续时间特征
                    duration_value = len(audio_data) / audio_fs

                    # 将特征添加到特征矩阵中
                    if pitch_features is None:
                        pitch_features = pitch_values
                    else:
                        pitch_features = np.vstack((pitch_features, pitch_values))

                    if timbre_features is None:
                        timbre_features = mfccs
                    else:
                        timbre_features = np.vstack((timbre_features, mfccs))

                    if loudness_features is None:
                        loudness_features = rms_values
                    else:
                        loudness_features = np.vstack((loudness_features, rms_values))

                    if duration_features is None:
                        duration_features = np.full((mfccs.shape[0], 1), duration_value)
                    else:
                        duration_features = np.vstack((duration_features, np.full((mfccs.shape[0], 1), duration_value)))

                    # 将情感标签添加到标签向量中
                    emotion_label = emotion_index
                    if labels is None:
                        labels = np.full((len(pitch_values), 1), emotion_label)
                    else:
                        labels = np.vstack((labels, np.full((len(pitch_values), 1), emotion_label)))

                # 转换为NumPy数组
                pitch_features = np.array(pitch_features)
                timbre_features = np.array(timbre_features)
                loudness_features = np.array(loudness_features)
                duration_features = np.array(duration_features)
                labels = np.array(labels)

                # 确保特征矩阵和标签向量的长度一致
                min_length = min(
                    [len(pitch_features), len(timbre_features), len(loudness_features), len(duration_features),
                     len(labels)])
                pitch_features = pitch_features[:min_length, :]
                timbre_features = timbre_features[:min_length, :]
                loudness_features = loudness_features[:min_length, :]
                duration_features = duration_features[:min_length, :]
                labels = labels[:min_length]

                # 保存特征矩阵和标签向量
                np.save('pitch_features_labels.npy', np.hstack((pitch_features, labels)))
                np.save('timbre_features_labels.npy', np.hstack((timbre_features, labels)))
                np.save('loudness_features_labels.npy', np.hstack((loudness_features, labels)))
                np.save('duration_features_labels.npy', np.hstack((duration_features, labels)))
