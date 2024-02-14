import os.path

import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AdamW
import numpy as np

from audio_dataset import AudioDataset
from data_process import data_preparation
from model import AudioClassifier


NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct_speaker = 0
        correct_label = 0
        total_samples = 0
        for i, batch in enumerate(train_loader):

            waveforms = batch['audio'].to(DEVICE)
            speakers = torch.from_numpy(np.array(batch['speaker']))
            labels = torch.from_numpy(np.array(batch['label']))

            # print(waveforms.squeeze(1).input_values.shape)

            optimizer.zero_grad()

            # loss
            speaker_logits, label_logits = model(waveforms)
            loss_speaker = criterion(speaker_logits, speakers)
            loss_label = criterion(label_logits, labels)
            total_loss = loss_speaker + loss_label
            running_loss += total_loss.item()

            # accuracy
            _, predicted_speakers = torch.max(speaker_logits, 1)
            _, predicted_labels = torch.max(label_logits, 1)
            total_samples += speakers.size(0)
            correct_speaker += (predicted_speakers == speakers).sum().item()
            correct_label += (predicted_labels == labels).sum().item()

            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss}")
        accuracy_speaker = correct_speaker / total_samples
        accuracy_label = correct_label / total_samples
        print(f"Training Accuracy - Speaker: {accuracy_speaker}, Label: {accuracy_label}")


# Example evaluation function
def evaluate(model, val_loader, model_save_path_):
    model.eval()
    correct_speaker = 0
    correct_label = 0
    total_samples = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            waveforms = batch['audio'].to(DEVICE)
            speakers = torch.from_numpy(np.array(batch['speaker']))
            labels = torch.from_numpy(np.array(batch['label']))

            speaker_logits, label_logits = model(waveforms)
            loss_speaker = criterion(speaker_logits, speakers)
            loss_label = criterion(label_logits, labels)
            running_loss = loss_speaker.item() + loss_label.item()

            _, predicted_speakers = torch.max(speaker_logits, 1)
            _, predicted_labels = torch.max(label_logits, 1)

            total_samples += speakers.size(0)
            correct_speaker += (predicted_speakers == speakers).sum().item()
            correct_label += (predicted_labels == labels).sum().item()

    print(f"Test Loss: {running_loss}")
    accuracy_speaker = correct_speaker / total_samples
    accuracy_label = correct_label / total_samples
    print(f"Test Accuracy - Speaker: {accuracy_speaker}, Label: {accuracy_label}")
    torch.save(model.state_dict(), os.path.join(model_save_path_, "model.pth"))


# Example usage
if __name__ == "__main__":

    csv_path = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset\\meta.csv'
    root_dir = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset'
    model_save_path = "C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification"
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    train_data_df, val_data_df, num_speakers, max_length, name_to_number_map = data_preparation(csv_path, root_dir)

    print("The max length of audio is: ", max_length)

    max_audio_length = 5000  # 200000
    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_data_df, root_dir, processor, max_audio_length)
    val_dataset = AudioDataset(val_data_df, root_dir, processor, max_audio_length)

    # for i in range(5):
    #     sample = train_dataset[i]
    #     print(i, sample[0].input_values.shape, sample[1][0], sample[1][1])
    # exit(0)

    batch_size = 4  # 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = AudioClassifier(num_speakers, 2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the model
    evaluate(model, val_loader, model_save_path)
