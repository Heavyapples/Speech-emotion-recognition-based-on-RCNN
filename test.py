import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# 定义情感列表
emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# 加载特征矩阵和标签向量
pitch_features_labels = np.load('pitch_features_labels.npy')
timbre_features_labels = np.load('timbre_features_labels.npy')
loudness_features_labels = np.load('loudness_features_labels.npy')
duration_features_labels = np.load('duration_features_labels.npy')

# 提取特征和标签
pitch_features, pitch_labels = pitch_features_labels[:, :-1], pitch_features_labels[:, -1]
timbre_features, timbre_labels = timbre_features_labels[:, :-1], timbre_features_labels[:, -1]
loudness_features, loudness_labels = loudness_features_labels[:, :-1], loudness_features_labels[:, -1]
duration_features, duration_labels = duration_features_labels[:, :-1], duration_features_labels[:, -1]

# 将特征堆叠在一起
stacked_features = np.hstack((pitch_features, timbre_features, loudness_features, duration_features))

# 数据预处理：将特征数据转换为适合CRNN的形状
n_features = stacked_features.shape[1]
all_data = stacked_features.reshape(-1, n_features, 1)

# 将标签转换为分类形式
num_classes = len(emotions)
all_labels = to_categorical(pitch_labels, num_classes)

# 加载训练好的模型
model_path = 'C:/Users/13729/PycharmProjects/mood/models/emotion_recognition_crnn_epoch067.h5'
trained_model = load_model(model_path)

# 评估模型在整个数据集上的准确率
_, accuracy = trained_model.evaluate(all_data, all_labels, batch_size=32)
print(f"模型在整个数据集上的准确率为：{accuracy * 100:.2f}%")
