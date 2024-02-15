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
from utils import equal_error_rate
import logging

torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

MAX_AUDIO_LENGTH = 120000  # 200000
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_CLASSES = 2

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(message)s')


# Training loop
def train(model_, train_loader_, val_loader_, criterion_classification, criterion_identification,
          optimizer, save_path, num_epochs=10, early_stop_threshold=3):
    highest_acc_speaker = 0.0
    highest_acc_label = 0.0
    lower_than_best_num = 0
    for epoch in tqdm(range(num_epochs)):
        model_.train()
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
        for i, batch in enumerate(train_loader_):
            waveforms = batch['audio'].to(DEVICE)
            speakers = torch.from_numpy(np.array(batch['speaker'])).to(DEVICE)
            labels = torch.from_numpy(np.array(batch['label'])).to(DEVICE)

            optimizer.zero_grad()

            # loss
            speaker_logits, label_logits = model_(waveforms)
            loss_speaker = criterion_classification(speaker_logits, speakers)
            loss_label = criterion_identification(label_logits.squeeze(), labels.float())
            total_loss = loss_speaker + 0.2 * loss_label
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

        # Loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Total Loss: {running_loss / total_samples}")
        print(f"Speaker Loss: {running_loss_speaker / total_samples},"
              f" Label Loss: {running_loss_label / total_samples}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Total Loss: {running_loss / total_samples}")
        logging.info(
            f"Speaker Loss: {running_loss_speaker / total_samples}, Label Loss: {running_loss_label / total_samples}")

        # Accuracy
        accuracy_speaker = correct_speaker / total_samples
        accuracy_label = correct_label / total_samples
        print(f"Training Accuracy - Speaker: {accuracy_speaker}, Label: {accuracy_label}")
        logging.info(f"Training Accuracy - Speaker: {accuracy_speaker}, Label: {accuracy_label}")

        # Calculate F1 score
        f1_speaker = f1_score(all_speakers, all_predicted_speakers, average='weighted')
        f1_label = f1_score(all_labels, all_predicted_labels, average='weighted')
        print(f"Training F1 Score - Speaker: {f1_speaker}, Label: {f1_label}")
        logging.info(f"Training F1 Score - Speaker: {f1_speaker}, Label: {f1_label}")

        # Calculate EER (equal error rate), only for binary classification task
        eer = equal_error_rate(all_labels, all_predicted_labels)
        print(f"Training EER: ", eer)
        print("")
        logging.info(f"Training EER: {eer}")
        logging.info("")

        # evaluate
        eval_speaker_acc, eval_label_acc = evaluate(model_, val_loader_,
                                                    criterion_classification, criterion_identification, "Validation")

        # Update the best value, and save the best model
        if eval_speaker_acc > highest_acc_speaker:
            highest_acc_speaker = eval_speaker_acc
            torch.save(model_.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            lower_than_best_num += 1

        if lower_than_best_num >= 3:
            print(f"Early stop. Total training epochs: {epoch + 1}")
            logging.info(f"Early stop. Total training epochs: {epoch + 1}")
            break


# Example evaluation function
def evaluate(model_eval, val_or_test_loader, criterion_classification, criterion_identification, eval_or_test):
    model_eval.eval()
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
    with torch.no_grad():
        for i, batch in enumerate(val_or_test_loader):
            waveforms = batch['audio'].to(DEVICE)
            speakers = torch.from_numpy(np.array(batch['speaker'])).to(DEVICE)
            labels = torch.from_numpy(np.array(batch['label'])).to(DEVICE)

            speaker_logits, label_logits = model_eval(waveforms)
            loss_speaker = criterion_classification(speaker_logits, speakers)
            loss_label = criterion_identification(label_logits.squeeze(), labels.float())
            running_loss = loss_speaker.item() + loss_label.item()
            running_loss_speaker += loss_speaker.item()
            running_loss_label += loss_label.item()

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

    # Loss
    print(f"{eval_or_test} Total Loss: {running_loss / total_samples}")
    print(f"Speaker Loss: {running_loss_speaker / total_samples},"
          f" Label Loss: {running_loss_label / total_samples}")
    logging.info(f"{eval_or_test} Total Loss: {running_loss / total_samples}")
    logging.info(
        f"Speaker Loss: {running_loss_speaker / total_samples}, Label Loss: {running_loss_label / total_samples}")

    # Accuracy
    accuracy_speaker = correct_speaker / total_samples
    accuracy_label = correct_label / total_samples
    print(f"{eval_or_test} Accuracy - Speaker: {accuracy_speaker}, Label: {accuracy_label}")
    logging.info(f"{eval_or_test} Accuracy - Speaker: {accuracy_speaker}, Label: {accuracy_label}")

    # Calculate F1 score
    f1_speaker = f1_score(all_speakers, all_predicted_speakers, average='weighted')
    f1_label = f1_score(all_labels, all_predicted_labels, average='weighted')
    print(f"{eval_or_test} F1 Score - Speaker: {f1_speaker}, Label: {f1_label}")
    logging.info(f"{eval_or_test} F1 Score - Speaker: {f1_speaker}, Label: {f1_label}")

    # Calculate EER (equal error rate), only for binary classification task
    eer = equal_error_rate(all_labels, all_predicted_labels)
    print(f"{eval_or_test} EER: ", eer)
    print("")
    logging.info(f"{eval_or_test} EER: {eer}")
    logging.info("")

    if eval_or_test == "Validation":
        return accuracy_speaker, accuracy_label
    # torch.save(model_eval.state_dict(), os.path.join(model_save_path_, "model.pth"))


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

    # csv_path, root_dir, model_save_path = local_run()
    csv_path, root_dir, model_save_path = remote_run()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    print("Device: ", DEVICE)
    print("The cuda is available: ", torch.cuda.is_available())

    train_data_df, val_data_df, test_data_df, num_speakers, max_length, name_to_number_map = \
        data_preparation(csv_path, root_dir)

    print("The max length of audio is: ", max_length)

    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_data_df, root_dir, processor, MAX_AUDIO_LENGTH)
    val_dataset = AudioDataset(val_data_df, root_dir, processor, MAX_AUDIO_LENGTH)
    test_dataset = AudioDataset(test_data_df, root_dir, processor, MAX_AUDIO_LENGTH)

    # for i in range(len(val_dataset)):
    #     sample = val_dataset[i]
    #     print(i, sample['speaker'], sample['label'])
    # exit(0)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = AudioClassifier(num_speakers, 1).to(DEVICE)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    adam_optimizer = AdamW(model.parameters(), lr=1e-5)

    # Train the model
    train(model, train_loader, val_loader, criterion1, criterion2,
          adam_optimizer, model_save_path, num_epochs=NUM_EPOCHS)

    # Test the model
    evaluate(model, test_loader, criterion1, criterion2, "Test")
