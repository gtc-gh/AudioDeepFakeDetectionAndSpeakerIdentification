import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
from datasets import load_dataset, load_metric
from transformers import AutoConfig, Wav2Vec2Processor


model_name_or_path = "lighteternal/wav2vec2-large-xlsr-53-greek"
pooling_mode = "mean"


def print_labels(df):
    print("Speakers: ", len(df['speaker'].unique()))  # we have 54 speakers


def data_preparation(csv_path_):
    """
    This function is used to prepare the data.
    :param csv_path_: the root for whole dataset
    :return: training dataset, evaluation dataset, the number of speakers
    """
    audio_csv_file_ = pd.read_csv(csv_path_)
    audio_csv_file_['label'] = audio_csv_file_['label'].map({'spoof': 0, 'bona-fide': 1})
    num_classes_ = len(audio_csv_file_['speaker'].unique())
    print(f"We have {num_classes_} different speakers.")

    # split the data
    audio_files = audio_csv_file_['file'].tolist()
    speakers = audio_csv_file_['speaker'].tolist()
    labels = audio_csv_file_['label'].tolist()

    train_audio, test_audio = train_test_split(audio_files, test_size=0.2, random_state=42)

    train_speaker, test_speaker = train_test_split(speakers, test_size=0.2, random_state=42)

    train_label, test_label = train_test_split(labels, test_size=0.2, random_state=42)

    train_data_ = {'file': train_audio, 'speaker': train_speaker, 'label': train_label}
    test_data_ = {'file': test_audio, 'speaker': test_speaker, 'label': test_label}

    df_train = pd.DataFrame(train_data_)
    df_test = pd.DataFrame(test_data_)

    save_path = os.path.sep.join(csv_path.split(os.path.sep)[:-1])
    train_path = os.path.join(save_path, "train.csv")
    test_path = os.path.join(save_path, "test.csv")
    df_train.to_csv(train_path, sep="\t", encoding="utf-8", index=False)
    df_test.to_csv(test_path, sep="\t", encoding="utf-8", index=False)

    print("Train data shape: ", df_train.shape)
    print("Test data shape: ", df_test.shape)

    # prepare data for training
    data_files = {
        "train": train_path,
        "validation": test_path
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    print("Train dataset: ", train_dataset)
    print("Test dataset: ", eval_dataset)

    return train_dataset, eval_dataset, num_classes_


if __name__ == "__main__":

    csv_path = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset\\meta.csv'
    root_dir = 'C:\\Users\\gtc\\Desktop\\AudioDeepFakeDetectionAndSpeakerIdentification\\dataset'

    train_dataset, eval_dataset, num_classes = data_preparation(csv_path)

    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)





