import numpy as np
import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2Processor


class AudioDataset(Dataset):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the audio files.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    def __init__(self, audio_metadata, root_dir, processor, max_length):
        self.audio_metadata = audio_metadata
        self.root_dir = root_dir
        self.processor = processor
        self.max_audio_length = max_length

    def __len__(self):
        return len(self.audio_metadata)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.audio_metadata.iloc[idx, 0])
        waveform, sample_rate = librosa.load(audio_path)
        waveform_16000 = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

        # Pad or truncate audio to the desired length
        if len(waveform_16000) > self.max_audio_length:
            waveform_16000_padded = waveform_16000[:self.max_audio_length]
        else:
            padding = self.max_audio_length - len(waveform_16000)
            waveform_16000_padded = np.pad(waveform_16000, (0, padding), mode='constant')

        input_features = self.processor(waveform_16000_padded, return_tensors="pt",
                                        padding=True, sampling_rate=16000).input_values.squeeze()
        # print(input_features.shape)

        # print(waveform_16000_padded.shape, sample_rate)

        # audio_features = self.processor(audio_raw, return_tensors="pt", padding="max_length",
        #                                 max_length=self.max_length, sampling_rate=16000)

        speaker = self.audio_metadata.iloc[idx, 1]
        label = self.audio_metadata.iloc[idx, 2]

        # print(audio_features.input_values.shape)  # torch.Size([1, 399856])

        # audio_tuple = (input_features, (speaker, label))
        audio_dic = {'audio': input_features,
                     'speaker': speaker,
                     'label': label}

        return audio_dic
