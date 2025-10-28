import sys
import os
from pathlib import Path
import requests
import zipfile
import io

import albumentations as A
# import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np 
import pydicom
import nibabel as nib
import skimage
from VITools.phantom import resize



def apply_window(image, center, width):
    '''taken directly from https://github.com/darraghdog/rsna'''
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def apply_window_policy(image):
    '''taken directly from https://github.com/darraghdog/rsna'''
    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)
    return image


def classify_images(volume, options, model_path, device):
    """Receive input volume of dimensions [num_images, 3, 480, 480] and
    performs classification with pre-trained models"""

    num_images = volume.shape[0]
    if options['verbose']: print('Starting classification')

    # Load and set up classifier
    model_path = Path(model_path)
    classifier_model = torch.load(model_path, weights_only=False, map_location=device) # Load model arch
    classifier_model.fc = nn.Linear(2048, 6) # 6 classes
    classifier_model.to(device)
    # Model was trained with DDP so need this even with 1 GPU
    classifier_model = nn.DataParallel(classifier_model, device_ids=list(range(1)), output_device=device)
    for param in classifier_model.parameters():
        param.requires_grad = False
    classifier_model.load_state_dict(torch.load(next(model_path.parent.glob('model_*.bin')),  map_location=device)) # Load weights

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
  
        def forward(self, x):
            return x

    # Extract embedding layers since that's what we need for the LSTM
    classifier_model.module.fc = Identity()
    classifier_model.eval()
    embeddings = np.zeros((num_images, 2048))
    for idx in range(num_images):
        slice = volume[idx, :, :, :].unsqueeze(0).to(device) # maintain first dimension
        out = classifier_model(slice)
        embeddings[idx, :] = out.detach().cpu().numpy().astype(np.float32)

    # Pad out the  embeddings
    lag = np.zeros(embeddings.shape)
    lead = np.zeros(embeddings.shape)
    lag[1:] = embeddings[1:]-embeddings[:-1]
    lead[:-1] = embeddings[:-1]-embeddings[1:]
    embeddings = np.concatenate((embeddings, lag, lead), -1)
    embeddings = torch.from_numpy(embeddings).unsqueeze(0).to(device, dtype=torch.float)

    # INITIALIZE CLASSES FOR LSTM MODEL
    class SpatialDropout(nn.Dropout2d):
        def forward(self, x):
            x = x.unsqueeze(2)    # (N, T, 1, K)
            x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
            x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
            x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
            x = x.squeeze(2)  # (N, T, K)
            return x

    class NeuralNet(nn.Module):
        def __init__(self, embed_size=2048*3, LSTM_UNITS=64, DO = 0.3):
            super(NeuralNet, self).__init__()

            self.embedding_dropout = SpatialDropout(0.0) #DO)

            self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
            self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

            self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
            self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

            self.linear = nn.Linear(LSTM_UNITS*2, 6)

        def forward(self, x, lengths=None):
            h_embedding = x

            h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)

            h_lstm1, _ = self.lstm1(h_embedding)
            h_lstm2, _ = self.lstm2(h_lstm1)

            h_conc_linear1  = F.relu(self.linear1(h_lstm1))
            h_conc_linear2  = F.relu(self.linear2(h_lstm2))

            hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd

            output = self.linear(hidden)

            return output

    # Create model 
    lstm_model = NeuralNet(LSTM_UNITS=2048, DO = 0.3)
    lstm_model = lstm_model.to(device)
    lstm_model.load_state_dict(torch.load(next(model_path.parent.glob('lstm_*.bin')),  map_location=device))
    for param in lstm_model.parameters():
        param.requires_grad = False

    lstm_model.eval()
    if options['verbose']: print('Evaluating embeddings')

    slice_by_slice = False # False is the original implementation for the grand challenge submission
    if slice_by_slice:
        values = np.zeros(shape=(embeddings.shape[1], 6))
        for i in range(embeddings.shape[1]):
            logits = lstm_model(embeddings[:, i, :].unsqueeze(1))
            logits = logits.view(-1, 6)
            temp = torch.sigmoid(logits).detach().cpu().numpy()
            if options['verbose']: print(temp[0][5])
            values[i, :] = temp
    else:   
        logits = lstm_model(embeddings)
        logits = logits.view(-1, 6)
        values = torch.sigmoid(logits).detach().cpu().numpy()

        if options['verbose']:
            for i in range(values.shape[0]):
                print('slice ' + str(i) + ': ' + str(values[i, -1]))
    return values


def predict_image(image, model, device='cuda'):
    # image preparation lifted from prepare_images (intended for .dcm and .nii)
    labels = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'Any']
    mean_img = [0.22363983, 0.18190407, 0.2523437 ]
    std_img = [0.32451536, 0.2956294,  0.31335256]
    transform = A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0)

    image = apply_window_policy(image)
    image -= image.min((0,1))
    image = (255*image).astype(np.uint8)
    #cv2.imwrite('test.jpg', image)
    image = resize(image, (480, 480)) # use Monai instead or VITools.resize
    result = transform(image=image)
    image = torch.from_numpy(result["image"])
    image = torch.permute(image, (2, 1, 0)).unsqueeze(0)


    # classify images
    options = {'verbose': False}
    output = classify_images(image, options, model, device=device)
    return dict(zip(labels, output[0]))


def download_and_unzip(url, extract_to='.'):
    """
    Downloads a zip file from a given URL and extracts its contents.

    Args:
        url (str): The URL of the zip file.
        extract_to (str): The directory where the contents should be extracted.
                          Defaults to the current directory.
    """
    try:
        # Download the zip file
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Create a BytesIO object from the downloaded content
        zip_file_in_memory = io.BytesIO(response.content)

        # Open the zip file and extract its contents
        with zipfile.ZipFile(zip_file_in_memory, 'r') as zip_ref:
            print(f"Extracting contents to {os.path.abspath(extract_to)}...")
            zip_ref.extractall(extract_to)
        print("Download and extraction complete.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")