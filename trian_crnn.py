import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

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

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(stacked_features, pitch_labels, test_size=0.2, random_state=42)

# 数据预处理：将特征数据转换为适合CRNN的形状
n_features = stacked_features.shape[1]
X_train = X_train.reshape(-1, n_features, 1)
X_test = X_test.reshape(-1, n_features, 1)

# 将标签转换为分类形式
num_classes = len(emotions)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 构建CRNN模型
model = Sequential()

# 卷积层
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(n_features, 1)))

# 循环层
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 加载已训练的模型
# model_path = 'emotion_recognition_crnn_epoch030.h5'
# trained_model = load_model(model_path)

# 创建回调每轮都保存模型
checkpoint = ModelCheckpoint('emotion_recognition_crnn_epoch{epoch:03d}.h5', save_freq='epoch')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# 从第30轮开始训练模型
# initial_epoch = 30
# trained_model.fit(X_train, y_train, epochs=100, initial_epoch=initial_epoch, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# 保存模型
model.save('emotion_recognition_crnn.h5')
