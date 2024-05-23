import os
import glob
import moviepy.editor as mp
from scipy.io import wavfile

# 设置文件夹路径
root_path = 'E:/代码接单/rcnn语音情感识别/project2_database/enterface database'
subject_folders = glob.glob(os.path.join(root_path, 'subject *'))
emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# 遍历所有志愿者文件夹
for subject_folder in subject_folders:

    # 遍历所有情感文件夹
    for emotion in emotions:
        emotion_folder = os.path.join(subject_folder, emotion)

        # 遍历所有句子文件夹
        for sentence_index in range(1, 6):
            sentence_folder = os.path.join(emotion_folder, f'sentence {sentence_index}')

            # 检查句子文件夹是否存在
            if os.path.isdir(sentence_folder):
                avi_files = glob.glob(os.path.join(sentence_folder, '*.avi'))

                # 检查AVI文件是否存在
                if avi_files:
                    avi_file_path = avi_files[0]

                    # 转换视频文件为音频文件
                    video = mp.VideoFileClip(avi_file_path)
                    audio = video.audio
                    audio_data = audio.to_soundarray()
                    audio_fs = audio.fps

                    # 保存音频文件为WAV格式
                    wav_file_path = os.path.join(sentence_folder,
                                                 f'{os.path.splitext(os.path.basename(avi_file_path))[0]}.wav')
                    wavfile.write(wav_file_path, audio_fs, audio_data)
