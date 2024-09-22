import os
import pdb
import yaml
import torch
import random
import librosa
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from einops import rearrange
from einops.layers.torch import Rearrange


def process_data():
    train_audio_folder = "train_mp3s"
    label_file = "train_label.txt"

    # train_audio_folder = "/content/drive/MyDrive/ML_final/train_mp3s_10"
    # label_file = "/content/drive/MyDrive/ML_final/train_label.txt"

    # train_audio_files = [os.path.join(train_audio_folder, file) for file in os.listdir(train_audio_folder) if file.endswith(".mp3")]

    train_audio_files = [f'train_mp3s/{i}.mp3' for i in range(11886)]

    # Get all labels, store in a list labels:[]
    with open(label_file, "r") as f:
        labels = f.readlines()
    labels = [int(label.strip()) for label in labels]

    # Ensure number of audio files == number of labels
    assert len(train_audio_files) == len(labels), "Number of audio files does not match number of labels"

    # Create empty list to store audio and corresponding labels
    audio_lst = []
    label_lst = []

    # Load audio files
    for train_audio_file, label in tqdm(zip(train_audio_files, labels)):
        # Load audio file, Resample audio to 16000 Hz, Convert to mono if necessary
        train_audio, sr = librosa.load(train_audio_file, sr=None)
        train_audio = librosa.resample(train_audio, orig_sr=sr, target_sr=16000)

        if len(train_audio.shape) > 1:
            train_audio = np.mean(train_audio, axis=1)

        # Append audio and label to lists
        audio_lst.append(train_audio)
        label_lst.append(label)

    # Convert audio and label lists into arrays
    audio_arr = np.array(audio_lst)
    label_arr = np.array(label_lst)

    # Split the data into training and validation sets
    train_size = 10000
    val_size = len(audio_lst) - train_size

    # Convert audio array to PyTorch tensor and move to GPU if available
    audio_tensor = torch.tensor(audio_arr, dtype=torch.float32)
    train_audio_tensor = audio_tensor[:train_size]
    val_audio_tensor = audio_tensor[train_size:]

    label_tensor = torch.tensor(label_arr, dtype=torch.float32)
    train_label_tensor = label_tensor[:train_size]
    val_label_tensor = label_tensor[train_size:]

    torch.save(audio_tensor, 'train.pt')
    torch.save(label_tensor, 'train_label.pt')


def preprocess():
    # Apply to our dataset
    train_audio_files = [f'train_mp3s/{i}.mp3' for i in range(11886)]

    # Get all labels, store in a list labels:[]
    with open('train_label.txt', "r") as f:
        labels = f.readlines()
    labels = [int(label.strip()) for label in labels]

    # Ensure number of audio files == number of labels
    assert len(train_audio_files) == len(labels), "Number of audio files does not match number of labels"

    spec = True

    if spec:
        # Create empty list to store audio and corresponding labels
        spectrogram_lst = []
        label_lst = []

        # Load audio files
        for train_audio_file, label in tqdm(zip(train_audio_files, labels)):
            spectrogram = audio_to_spectrogram(train_audio_file)
            spectrogram_lst.append(spectrogram)
            label_lst.append(label)

        # Convert audio and label lists into arrays
        spectrogram_arr = np.array(spectrogram_lst)
        label_arr = np.array(label_lst)

        audio_tensor = torch.tensor(spectrogram_arr, dtype=torch.float32)
        label_tensor = torch.tensor(label_arr, dtype=torch.float32)
        
        torch.save(audio_tensor, 'train_data_spec.pt')
        torch.save(label_tensor, 'train_label_spec.pt')
    else:
        # save the original loaded audio files
        # Create empty list to store audio and corresponding labels
        data_lst = []
        label_lst = []

        # Load audio files
        for train_audio_file, label in tqdm(zip(train_audio_files, labels)):
            audio, _ = librosa.load(train_audio_file, sr=None)
            data_lst.append(audio)

        # Convert audio and label lists into arrays
        data_arr = np.array(data_lst)

        # Convert audio array to PyTorch tensor and move to GPU if available
        audio_tensor = torch.tensor(data_arr, dtype=torch.float32)
        
        torch.save(audio_tensor, 'train_data.pt')


def preprocess_test():
    # Apply to our dataset
    train_audio_files = [f'test_mp3s/{i}.mp3' for i in range(2447)]

    # Create empty list to store audio and corresponding labels
    spectrogram_lst = []
    label_lst = []

    # Load audio files
    for train_audio_file in tqdm(train_audio_files):
        spectrogram = audio_to_spectrogram(train_audio_file)
        spectrogram_lst.append(spectrogram)

    # Convert audio and label lists into arrays
    spectrogram_arr = np.array(spectrogram_lst)

    # Convert audio array to PyTorch tensor and move to GPU if available
    audio_tensor = torch.tensor(spectrogram_arr, dtype=torch.float32)

    torch.save(audio_tensor, 'augment_data/test_data_spec.pt')


class Echo(object):
    def __init__(self, echo_strength=0.05):
        self.echo_strength = echo_strength

    def __call__(self, audio_tensor):
        if random.random() < float(os.environ['AUGMENT_PROB']):
            audio = audio_tensor.numpy()
            echoed_audio = audio + self.echo_strength * np.roll(audio, 16000)

            return torch.from_numpy(echoed_audio)

        return audio_tensor


class WhiteNoise(object):
    def __init__(self, noise_level=0.005):
        self.noise_level = noise_level

    def __call__(self, audio_tensor):
        if random.random() < float(os.environ['AUGMENT_PROB']):
            noise = torch.randn_like(audio_tensor) * self.noise_level
            audio_with_noise = audio_tensor + noise

            return audio_with_noise

        return audio_tensor


class SpeedUp(object):
    def __init__(self):
        pass

    def __call__(self, audio_tensor):
        if random.random() < float(os.environ['AUGMENT_PROB']):
            speed_factor = random.uniform(0.95, 1.05)

            audio = audio_tensor.numpy()
            new_length = int(len(audio) / speed_factor)
            stretched_audio = librosa.effects.time_stretch(audio, rate=speed_factor)

            stretched_audio = librosa.resample(stretched_audio, orig_sr=stretched_audio.shape[0], target_sr=audio.shape[0])

            if len(stretched_audio) < new_length:
                stretched_audio = np.pad(stretched_audio, (0, len(audio) - len(stretched_audio)), mode='constant')

            return torch.from_numpy(stretched_audio)
        
        return audio_tensor


class IncreseVolume(object):
    def __init__(self, min_vol, max_vol):
        self.min_vol = min_vol
        self.max_vol = max_vol

    def __call__(self, audio_tensor):
        if random.random() < float(os.environ['AUGMENT_PROB']):
            volume_factor = random.uniform(self.min_vol, self.max_vol)
            perturb_audio_tensor = audio_tensor * volume_factor

            return perturb_audio_tensor

        return audio_tensor


class AdjustPitch(object):
    def __init__(self, pitch_factor=0.05):
        self.pitch_factor = pitch_factor

    def __call__(self, audio_tensor):
        if random.random() < float(os.environ['AUGMENT_PROB']):
            audio = audio_tensor.numpy()
            pitch_audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=self.pitch_factor)

            return torch.from_numpy(pitch_audio)

        return audio_tensor


def pad_to_original_shape(cropped, original_shape):
    padded = torch.zeros(original_shape)
    padded[:cropped.shape[0], :cropped.shape[1]] = cropped
    return padded


def audio_to_spectrogram(audio):
    # Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio)

    # Convert to decibel scale
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Normalize the spectrogram
    spectrogram_db = librosa.util.normalize(spectrogram_db)

    return torch.tensor(spectrogram_db, dtype=torch.float32)


class TrainDatset(torch.utils.data.Dataset):
    def __init__(self):
        # [10686, 132300]
        # self.audio_tensor = torch.load('data/nosr_train_audio.pt')

        self.audio_files = [f'data/train_mp3s/{i}.mp3' for i in range(11886)][1200:]
        
        # [10686]
        # self.audio_label = torch.load('data/nosr_train_labels.pt')
        lines = []
        with open('data/train_label.txt', 'r') as file:
            lines = file.readlines()
        
        self.audio_label = [int(line.strip()) for line in lines]
        self.audio_label = torch.tensor(self.audio_label, dtype=torch.int64)[1200:]

        assert len(self.audio_files) == self.audio_label.shape[0], "Number of audio files does not match number of labels"

        self.transform = transforms.Compose([
            Echo(),
            WhiteNoise(),
            SpeedUp(),
            IncreseVolume(min_vol=0.85, max_vol=1.15),
            AdjustPitch(),
        ])

    def __len__(self):
        # return self.audio_tensor.shape[0]
        return len(self.audio_files)

    def __getitem__(self, idx):
        # audio = self.audio_tensor[idx]
        audio, _ = librosa.load(self.audio_files[idx], sr=None)
        audio = torch.tensor(audio, dtype=torch.float32)

        self.transform(audio)

        # convert to spectrogram
        spectrogram = audio_to_spectrogram(audio.numpy())

        org_shape = spectrogram.shape

        if random.random() < float(os.environ['AUGMENT_PROB']):
            crop_ratio = random.uniform(0.85, 0.99)
            crop = transforms.RandomCrop(
                (int(spectrogram.shape[0] * crop_ratio), int(spectrogram.shape[1] * crop_ratio))
            )

            spectrogram = crop(spectrogram)
        
            # pad the cropped spectrogram to original shape
            spectrogram = pad_to_original_shape(spectrogram, (org_shape[-2], org_shape[-1]))

        return spectrogram, self.audio_label[idx]


class ValDatset(torch.utils.data.Dataset):
    def __init__(self):
        # # [10686, 132300]
        # self.audio_tensor = torch.load('data/nosr_val_audio.pt')
        # # [10686]
        # self.audio_label = torch.load('data/nosr_val_labels.pt')

        self.audio_files = [f'data/train_mp3s/{i}.mp3' for i in range(11886)][:1200]
        
        # [10686]
        # self.audio_label = torch.load('data/nosr_train_labels.pt')
        lines = []
        with open('data/train_label.txt', 'r') as file:
            lines = file.readlines()
        
        self.audio_label = [int(line.strip()) for line in lines]
        self.audio_label = torch.tensor(self.audio_label, dtype=torch.int64)[:1200]

        assert len(self.audio_files) == self.audio_label.shape[0], "Number of audio files does not match number of labels"

    def __len__(self):
        # return self.audio_tensor.shape[0]
        return len(self.audio_files)

    def __getitem__(self, idx):
        # audio = self.audio_tensor[idx]
        audio, _ = librosa.load(self.audio_files[idx], sr=None)
        audio = torch.tensor(audio, dtype=torch.float32)

        # convert to spectrogram
        spectrogram = audio_to_spectrogram(audio.numpy())

        return spectrogram, self.audio_label[idx]


class TestDatset(torch.utils.data.Dataset):
    def __init__(self):
        # self.audio_tensor = torch.load('data/nosr_test_audio.pt')
        self.audio_files = [f'data/test_mp3s/{i}.mp3' for i in range(2447)]

    def __len__(self):
        # return self.audio_tensor.shape[0]
        return len(self.audio_files)

    def __getitem__(self, idx):
        # audio = self.audio_tensor[idx]
        audio, _ = librosa.load(self.audio_files[idx], sr=None)
        audio = torch.tensor(audio, dtype=torch.float32)

        # convert to spectrogram
        spectrogram = audio_to_spectrogram(audio.numpy())

        return spectrogram


if __name__ == '__main__':
    os.environ['AUGMENT_PROB'] = '1'

    trainset = TrainDatset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    valset = ValDatset()
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)

    testset = TestDatset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    for i, (audio, label) in enumerate(trainloader):
        # [32, 128, 259]
        print("Train laoder passed, audio shape:", audio.shape)
        break

    for i, (audio, label) in enumerate(valloader):
        print("Val loader passed, audio shape:", audio.shape)
        break

    for i, audio in enumerate(testloader):
        print("Test loader passed, audio shape:", audio.shape)
        break
