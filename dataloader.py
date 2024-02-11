import numpy as np
import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class AudioDataset(Dataset):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the audio files.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    def __init__(self, audio_metadata, root_dir, max_length, transform=None,
                 sr=2000, n_mfcc=13, n_fft=2048, hop_length=512):
        self.audio_metadata = audio_metadata
        self.root_dir = root_dir
        self.max_length = max_length
        self.transform = transform
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.audio_metadata)

    def __getitem__(self, idx):
        audio_name = os.path.join(self.root_dir, self.audio_metadata.iloc[idx, 0])
        y, sr = librosa.load(audio_name)

        # get the MFCC feature
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)

        # pad the mfcc features
        if self.max_length is not None:
            mfcc = self._pad_or_truncate_mfcc(mfcc, self.max_length)

        speaker = self.audio_metadata.iloc[idx, 1]
        label = self.audio_metadata.iloc[idx, 2]

        current_audio = {'audio': mfcc, 'speaker': speaker, "label": label}

        if self.transform:
            current_audio = self.transform(current_audio)

        return current_audio

    def _pad_or_truncate_mfcc(self, mfcc, max_length):
        if mfcc.shape[1] < max_length:
            pad_with = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_with)), mode="constant", constant_values=0)
        elif mfcc.shape[1] > max_length:
            mfcc = mfcc[:, :max_length]
        return mfcc



