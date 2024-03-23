import platform,os
import ffmpeg
import numpy as np
import wave
from scipy.signal import resample

def load_audio(file, sr):
    file = clean_path(file)
    with wave.open(file, 'rb') as wav_file:
        # 读取音频参数
        n_channels, sample_width, framerate, n_frames, _, _ = wav_file.getparams()
        
        # 读取音频数据
        frames = wav_file.readframes(n_frames)
        
        # 将音频数据转换为numpy数组
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width")
        
        audio = np.frombuffer(frames, dtype=dtype)
        
        # 转换为浮点数格式
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
        
        # 如果是多声道，转换为单声道
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        
        # 改变采样率
        num_samples = int(len(audio) * float(sr) / framerate)
        audio_resampled = resample(audio, num_samples)
        
    return audio_resampled


def clean_path(path_str):
    if platform.system() == 'Windows':
        path_str = path_str.replace('/', '\\')
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
