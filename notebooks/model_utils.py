import albumentations as A
import cv2
from pathlib import Path
import sys
import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np 
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import skimage

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


def prepare_images(filepath, options, ID, save_dir):
    """Receive input image (JPG, NifTI, DICOM) and 
    returns tensor for input into trained model
    Modified from: https://github.com/darraghdog/rsna"""
    
    mean_img = [0.22363983, 0.18190407, 0.2523437 ]
    std_img = [0.32451536, 0.2956294,  0.31335256]
    transform = A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0)

    if os.path.isdir(filepath):
        files = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
        files = sorted(files)

        if options['native_resolution']:
            volume = torch.zeros((len(files), 3, options['size'][0], options['size'][1]))
            for idx, file in enumerate(files):
                if Path(file).suffix == '.dcm':
                    try:
                        # Convert from dicom to three channels (option to save as jpg)
                        filenm = file.split('.')[0]
                        dicom = pydicom.dcmread(os.path.join(filepath, file))
                        image = dicom.pixel_array
                        
                        image = image * dicom.RescaleSlope + dicom.RescaleIntercept
                        
                        image = apply_window_policy(image)
                        image -= image.min((0,1))
                        image = (255*image).astype(np.uint8)
                        #image = np.rot90(image,k=2) # this may not be necessary, but rotating to same orientation as RSNA training data

                        # Resize and normalize
                        image = cv2.resize(image, (480, 480))
                        result = transform(image=image)

                        if options['save_jpg']:
                            cv2.imwrite(os.path.join(save_dir, filenm) + '.jpg', result["image"])

                        image = torch.from_numpy(result["image"])
                        image = torch.permute(image, (2, 1, 0)).unsqueeze(0)
                        
                        volume[idx, :, :, :] = image
                    except:
                        print('Failed to read and convert ' + file)
        else:
            slice_thickness = 5
            
            for idx, file in enumerate(files):
                if Path(file).suffix == '.dcm':
                    try:
                        filenm = file.split('.')[0]
                        dicom = pydicom.dcmread(os.path.join(filepath, file))
                        image = dicom.pixel_array

                        image = image * dicom.RescaleSlope + dicom.RescaleIntercept
                        if idx == 0:
                            dcm_volume = np.zeros((len(files), image.shape[0], image.shape[1]))
                        dcm_volume[idx, :, :] = image
                    except:
                        print('Failed to read and convert ' + file)

            new_volume = skimage.measure.block_reduce(dcm_volume, block_size=(slice_thickness,1,1), func=np.mean, cval=np.mean(dcm_volume))
            volume = torch.zeros((new_volume.shape[0], 3, options['size'][0], options['size'][1]))
            for i in range(new_volume.shape[0]):
                #try:
                image = new_volume[i, :, :]

                if options['save_dcm']:
                    ds = dicom
                    ds.PatientName = ds.PatientID = str(ID)
                    ds.PixelData = (image - dicom.RescaleIntercept).astype(np.int16)
                    ds.SliceThickness = 5
                    ds.save_as(os.path.join(save_dir, str(ID)) + '_' + str(i) + '_5mm.dcm')
                    
                image = apply_window_policy(image)
                image -= image.min((0,1))
                image = (255*image).astype(np.uint8)
                #image = np.rot90(image,k=2) # this may not be necessary, but rotating to same orientation as RSNA training data

                if options['save_jpg']:
                    cv2.imwrite(os.path.join(save_dir, str(ID)) + '_' + str(i) + '_new5mm.jpg', image)

                # Resize and normalize
                image = cv2.resize(image, (480, 480))
                result = transform(image=image)
                image = torch.from_numpy(result["image"])
                image = torch.permute(image, (2, 1, 0)).unsqueeze(0)
                
                volume[i, :, :, :] = image
                #except:
                    #print('Failed to read and convert ' + file)

            #nib.save(nib.Nifti1Image(new_volume.transpose(2, 1, 0), np.eye(4)), '/gpfs_projects/jayse.weaver/pedsilicoICH_10-09-2024_5mm/temp_minus5HU.nii')

        return volume, files
    
    else: # path must be a single image
        if Path(filepath).suffix == '.jpg':
            if options['verbose']: print('Loading and preprocessing .jpg file')
            if options['verbose']: print('Assuming channels have been properly set')
            img = cv2.imread(filepath)

            # Resize and normalize
            img = cv2.resize(img, (480, 480))
            result = transform(image=img)
            img = torch.from_numpy(result["image"])
            img = torch.permute(img, (2, 1, 0)).unsqueeze(0)

            volume = img
            
        elif Path(filepath).suffix == '.nii':
            if options['verbose']: print('Loading and preprocessing NifTI file')
            nifti_file = nib.load(filepath)
            nifti_img = nifti_file.get_fdata()

            volume = torch.zeros((nifti_img.shape[2], 3, options['size'][0], options['size'][1]))

            nifti_img = nifti_img * nifti_file.dataobj.slope + nifti_file.dataobj.inter

            for slice in range(nifti_img.shape[2]):
                image = apply_window_policy(nifti_img[:, :, slice])
                image -= image.min((0,1))
                image = (255*image).astype(np.uint8)
                image = np.rot90(image,k=1) # this may not be necessary, but rotating to same orientation as RSNA training data

                if slice == 16:
                    if options['save_jpg']:
                        cv2.imwrite(os.path.join(save_dir, ID + '_slice'+str(slice)) + '.jpg', image)

                # Resize and normalize
                image = cv2.resize(image, (480, 480))
                result = transform(image=image)
                image = torch.from_numpy(result["image"])
                image = torch.permute(image, (2, 1, 0)).unsqueeze(0)
                
                volume[slice, :, :, :] = image

        else:
            sys.exit('No appropriate extension')

    return volume, filepath

def classify_images(volume, options, device):
    """Receive input volume of dimensions [num_images, 3, 480, 480] and
    performs classification with pre-trained models"""

    num_images = volume.shape[0]
    if options['verbose']: print('Starting classification')
    
    # Load and set up classifier
    classifier_model = torch.load('/home/jayse.weaver/model_files/resnext101_32x8d_wsl_checkpoint.pth', weights_only=False, map_location=device) # Load model arch
    classifier_model.fc = nn.Linear(2048, 6) # 6 classes
    classifier_model.to(device)
    # Model was trained with DDP so need this even with 1 GPU
    classifier_model = nn.DataParallel(classifier_model, device_ids=list(range(1)), output_device=device)
    for param in classifier_model.parameters():
        param.requires_grad = False
    classifier_model.load_state_dict(torch.load('/home/jayse.weaver/model_files/model_480_epoch2_fold6.bin',  map_location=device)) # Load weights

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
    lstm_model.load_state_dict(torch.load('/home/jayse.weaver/model_files/lstm_gepoch2_lstmepoch11_fold6.bin',  map_location=device))
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
