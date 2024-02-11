import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataloader
from dataloader import AudioDataset
from sklearn.model_selection import train_test_split


max_length = 450
batch_size = 64
num_epochs = 10


def data_preparation(csv_path_):
    """
    This function is used to prepare the data.
    :param csv_path_: the root for whole dataset
    :return: train data, validation data, test data, the number of speaker classes
    """
    audio_csv_file_ = pd.read_csv(csv_path_)
    audio_csv_file_['label'] = audio_csv_file_['label'].map({'spoof': 0, 'bona-fide': 1})
    num_classes_ = len(audio_csv_file_['speaker'].unique())

    # split the data
    audio_files = audio_csv_file_['file'].tolist()
    speakers = audio_csv_file_['speaker'].tolist()
    labels = audio_csv_file_['label'].tolist()

    train_audio, test_audio = train_test_split(audio_files, test_size=0.3, random_state=42)
    test_audio, val_audio = train_test_split(test_audio, test_size=0.33, random_state=42)

    train_speaker, test_speaker = train_test_split(speakers, test_size=0.3, random_state=42)
    test_speaker, val_speaker = train_test_split(test_speaker, test_size=0.33, random_state=42)

    train_label, test_label = train_test_split(labels, test_size=0.3, random_state=42)
    test_label, val_label = train_test_split(test_label, test_size=0.33, random_state=42)

    train_data_ = {'file': train_audio, 'speaker': train_speaker, 'label': train_label}
    val_data_ = {'file': val_audio, 'speaker': val_speaker, 'label': val_label}
    test_data_ = {'file': test_audio, 'speaker': test_speaker, 'label': test_label}

    return pd.DataFrame(train_data_), pd.DataFrame(val_data_), pd.DataFrame(test_data_), num_classes_


def combined_loss(logits_classification, embeddings_identification,
                  labels_classification, labels_identification,
                  loss_function_classification, loss_function_identification,
                  theta_classification, theta_identification):
    loss_classification = loss_function_classification(logits_classification, labels_classification)
    loss_identification = loss_function_identification(embeddings_identification, labels_identification)
    return theta_classification * loss_classification + theta_identification * loss_identification


def train_process(train_dataloader, val_dataloader, test_dataloader,
                  speaker_num_classes, embedding_dim_identification=512):

    # load the pretrained model
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    # modify the top layer for multitask learning
    model.classification_head = nn.Linear(model.config.hidden_size, speaker_num_classes)
    model.identification_head = nn.Linear(model.config.hidden_size, embedding_dim_identification)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # define loss function
    classification_loss_function = nn.CrossEntropyLoss()
    identification_loss_function = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs)):
        for batch in train_dataloader:
            audio_features, classification_labels, identification_labels = batch
            print(audio_features.shape, classification_labels.shape, identification_labels.shape)
            outputs = model(audio_features)
            logits_classification = outputs.logits
            embedding_identification = outputs.last_hidden_state.mean(dim=1)

            loss = combined_loss(logits_classification, embedding_identification,
                                 classification_labels, identification_labels,
                                 classification_loss_function, identification_loss_function,
                                 1, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate the model
        val_loss, val_preds, val_labels = validate_model(model, val_dataloader,
                                                         classification_loss_function, identification_loss_function)

        # Calculate metrics for validation and test sets
        # For demonstration purposes, let's assume we're using accuracy as the metric
        val_accuracy = accuracy_score(np.argmax(val_preds, axis=1), val_labels)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Test the model
    test_loss, test_preds, test_labels = validate_model(model, test_dataloader,
                                                        classification_loss_function, identification_loss_function)
    test_accuracy = accuracy_score(np.argmax(test_preds, axis=1), test_labels)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_accuracy}")


def validate_model(model, dataloader, classification_loss_function, identification_loss_function):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            audio_features, classification_labels, identification_labels = batch
            outputs = model(audio_features)
            logits_classification = outputs.logits
            embeddings_identification = outputs.last_hidden_state.mean(dim=1)

            loss = combined_loss(logits_classification, embeddings_identification,
                                 classification_labels, identification_labels,
                                 classification_loss_function, identification_loss_function,
                                 1, 1)
            val_loss += loss.item()

            # Store predictions and labels for metrics calculation
            all_preds.append(logits_classification.detach().cpu().numpy())
            all_labels.append(classification_labels.cpu().numpy())

    val_loss /= len(dataloader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return val_loss, all_preds, all_labels


if __name__ == "__main__":

    csv_path = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset\\meta.csv'
    root_dir = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset'

    # get the csv file, and class number
    train_data, val_data, test_data, num_classes = data_preparation(csv_path)

    train_dataset = AudioDataset(train_data, root_dir, max_length)
    val_dataset = AudioDataset(val_data, root_dir, max_length)
    test_dataset = AudioDataset(test_data, root_dir, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_process(train_dataloader, val_dataloader, test_dataloader, num_classes)

    # for i in range(2):
    #     sample = train_dataset[i]
    #     print(i, sample['audio'].shape, sample['speaker'], sample['label'])

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


