import os.path

import librosa
import pandas as pd
from sklearn.model_selection import train_test_split


def data_preparation(csv_path_, root_dir_):
    """
    This function is used to prepare the data.
    :param csv_path_: the root for csv metadata, root_dir: the root for the whole dataset
    :return: training dataset, evaluation dataset, the number of speakers, name-number map
    """
    audio_csv_file_ = pd.read_csv(csv_path_)

    # cut the csv
    # audio_csv_file_ = audio_csv_file_[:200]

    audio_csv_file_['label'] = audio_csv_file_['label'].map({'spoof': 0, 'bona-fide': 1})

    # map names to numbers
    unique_names = audio_csv_file_['speaker'].unique()
    num_classes_ = len(unique_names)
    print(f"We have {num_classes_} different speakers.")
    name_to_number = {name: idx for idx, name in enumerate(unique_names)}
    audio_csv_file_['speaker'] = audio_csv_file_['speaker'].map(name_to_number)
    audio_csv_file_ = audio_csv_file_.sample(frac=1, random_state=42)

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

    df_train = pd.DataFrame(train_data_)
    df_val = pd.DataFrame(val_data_)
    df_test = pd.DataFrame(test_data_)

    # save_path = os.path.sep.join(csv_path.split(os.path.sep)[:-1])
    # train_path = os.path.join(save_path, "train.csv")
    # test_path = os.path.join(save_path, "test.csv")
    # df_train.to_csv(train_path, sep="\t", encoding="utf-8", index=False)
    # df_test.to_csv(test_path, sep="\t", encoding="utf-8", index=False)

    print("Train data shape: ", df_train.shape)
    print("Validation data shape: ", df_val.shape)
    print("Test data shape: ", df_test.shape)

    # prepare data for training
    # data_files = {
    #     "train": train_path,
    #     "validation": test_path
    # }
    # dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    # train_dataset = dataset['train']
    # eval_dataset = dataset['validation']
    #
    # print("Train dataset: ", train_dataset)
    # print("Test dataset: ", eval_dataset)

    # find the max_length of all input audios
    # max_duration = 0.0
    # for idx in range(len(audio_csv_file_)):
    #     cur_path = os.path.join(root_dir_, audio_csv_file_.iloc[idx, 0])
    #     audio_duration = librosa.get_duration(filename=cur_path)
    #     max_duration = max(max_duration, audio_duration)
    #
    # # Assuming sampling_rate is 16000 Hz, convert max_duration to max_length in samples
    # sampling_rate = 16000
    # max_length_ = int(max_duration * sampling_rate)
    max_length_ = 399856

    return df_train, df_val, df_test, num_classes_, max_length_, name_to_number
