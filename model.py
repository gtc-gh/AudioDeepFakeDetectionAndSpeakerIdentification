import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AdamW


# class ClassificationHead(nn.Module):
#     def __init__(self, input_features):
#         super(ClassificationHead, self).__init__()
#         self.dense = nn.Linear()


class AudioClassifier(nn.Module):
    def __init__(self, num_speakers, num_classes):
        super(AudioClassifier, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.speaker_classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(512, num_speakers)
        )

        self.label_classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_features):
        # input_features = self.processor(audio_raw, return_tensors="pt",
        #                                 padding=True, sampling_rate=16000)
        hidden_states = self.wav2vec2(input_features).last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)

        speaker_logits = self.speaker_classifier(pooled_output)
        label_logits = self.label_classifier(pooled_output)

        return speaker_logits, label_logits