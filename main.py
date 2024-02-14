import os.path

import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np

from audio_dataset import AudioDataset
from data_process import data_preparation
from model import AudioClassifier


torch.cuda.empty_cache()
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
MAX_AUDIO_LENGTH = 150000  # 200000
BATCH_SIZE = 8
NUM_EPOCHS = 5


# Training loop
def train(model, train_loader, criterion_classification, criterion_identification, optimizer, num_epochs=10):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        running_loss_speaker = 0.0
        running_loss_label = 0.0
        correct_speaker = 0
        correct_label = 0
        total_samples = 0
        all_predicted_speakers = []
        all_predicted_labels = []
        all_speakers = []
        all_labels = []
        for i, batch in enumerate(train_loader):

            waveforms = batch['audio'].to(DEVICE)
            speakers = torch.from_numpy(np.array(batch['speaker'])).to(DEVICE)
            labels = torch.from_numpy(np.array(batch['label'])).to(DEVICE)

            optimizer.zero_grad()

            # loss
            speaker_logits, label_logits = model(waveforms)
            loss_speaker = criterion_classification(speaker_logits, speakers)
            loss_label = criterion_identification(label_logits, labels)
            total_loss = loss_speaker + 0.3 * loss_label
            running_loss_speaker += loss_speaker.item()
            running_loss_label += loss_label.item()
            running_loss += total_loss.item()

            # accuracy
            _, predicted_speakers = torch.max(speaker_logits, 1)
            _, predicted_labels = torch.max(label_logits, 1)
            total_samples += speakers.size(0)
            correct_speaker += (predicted_speakers == speakers).sum().item()
            correct_label += (predicted_labels == labels).sum().item()

            # Collect predictions and ground truths for F1 score calculation
            all_predicted_speakers.extend(predicted_speakers.tolist())
            all_predicted_labels.extend(predicted_labels.tolist())
            all_speakers.extend(speakers.tolist())
            all_labels.extend(labels.tolist())

            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Total Loss: {running_loss / total_samples}")
        print(f"Speaker Loss: {running_loss_speaker / total_samples},"
              f" Label Loss: {running_loss_label / total_samples}")
        accuracy_speaker = correct_speaker / total_samples
        accuracy_label = correct_label / total_samples
        print(f"Training Accuracy - Speaker: {accuracy_speaker}, Label: {accuracy_label}")
        # Calculate F1 score
        f1_speaker = f1_score(all_speakers, all_predicted_speakers, average='weighted')
        f1_label = f1_score(all_labels, all_predicted_labels, average='weighted')
        print(f"Training F1 Score - Speaker: {f1_speaker}, Label: {f1_label}")
        print("")


# Example evaluation function
def evaluate(model, val_loader, criterion_classification, criterion_identification, model_save_path_):
    model.eval()
    correct_speaker = 0
    correct_label = 0
    total_samples = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            waveforms = batch['audio'].to(DEVICE)
            speakers = torch.from_numpy(np.array(batch['speaker'])).to(DEVICE)
            labels = torch.from_numpy(np.array(batch['label'])).to(DEVICE)

            speaker_logits, label_logits = model(waveforms)
            loss_speaker = criterion_classification(speaker_logits, speakers)
            loss_label = criterion_identification(label_logits, labels)
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


def local_run():
    csv_path_ = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset\\meta.csv'
    root_dir_ = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset'
    model_save_path_ = "C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification"
    return csv_path_, root_dir_, model_save_path_


def remote_run():
    csv_path__ = '/home/tianchenguo/audio_classification_identification/dataset/meta.csv'
    root_dir__ = '/home/tianchenguo/audio_classification_identification/dataset'
    model_save_path__ = '/home/tianchenguo/audio_classification_identification'
    return csv_path__, root_dir__, model_save_path__


if __name__ == "__main__":

    csv_path, root_dir, model_save_path = local_run()
    # csv_path, root_dir, model_save_path = remote_run()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    print("Device: ", DEVICE)
    print("The cuda is available: ", torch.cuda.is_available())

    train_data_df, val_data_df, num_speakers, max_length, name_to_number_map = data_preparation(csv_path, root_dir)

    print("The max length of audio is: ", max_length)

    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_data_df, root_dir, processor, MAX_AUDIO_LENGTH)
    val_dataset = AudioDataset(val_data_df, root_dir, processor, MAX_AUDIO_LENGTH)

    # for i in range(5):
    #     sample = train_dataset[i]
    #     print(i, sample[0].input_values.shape, sample[1][0], sample[1][1])
    # exit(0)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = AudioClassifier(num_speakers, 2).to(DEVICE)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Train the model
    train(model, train_loader, criterion1, criterion2, optimizer, num_epochs=NUM_EPOCHS)

    # Evaluate the model
    evaluate(model, val_loader, criterion1, criterion2, model_save_path)
