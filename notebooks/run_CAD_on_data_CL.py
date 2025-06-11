import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch
from dotenv import load_dotenv

from model_utils import prepare_images, classify_images

load_dotenv()

# Large datasets too cumbersome to run via notebook (run_CAD_on_data.ipynb)
options = {
    'size': (480, 480),
    'save_jpg': False,
    'save_dcm': False,
    'save_csv': True,
    'verbose': False,
    'native_resolution': False # synthetic data is 1 mm, "False" will reformat to 5 mm
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device ' + str(device))

dataset_path = Path(os.environ['DATASET_PATH'])  # Set this to your dataset path
dataset_names = ['manuscript_100_280mA_wME', 'manuscript_100_280mA_noME']

prepath = Path(os.environ['MODEL_PATH'])  # Set this to your model path
model_names = ['CAD_1', 'CAD_2', 'CAD_3']

save_dir = './temp_images/new/' # optional save directory if save_jpg or save_dcm = True

for model in model_names:
    print('Processing with model ' + str(model))
    model_path = prepath + model + '/'
    for dataset in dataset_names:
        print('Processing dataset: ' + str(dataset))
        path = dataset_path / dataset
        metadata = pd.concat([pd.read_csv(o) for o in path.rglob('metadata*.csv')], ignore_index=True)

        if options['save_csv']: metadata.to_csv(path / (str(dataset) + '_' + model + '_metadata.csv'))

        metadata_short = metadata.drop(metadata.columns[12:], axis=1)
        metadata_dropna = metadata_short.dropna()

        cases = sorted(os.listdir(path))
        cases = [case for case in cases if case.startswith('case')]

        # initialize empty lists for results CSV
        attenuation_list = []
        volume_list = []
        type_list = []
        labels_list = []
        preds_list = []

        if options['native_resolution']:
            print('Processing in native resolution')
        else:
            print('Processing in 5 mm resolution')

        max_count = len(cases)
        description = 'Processing ' + str(len(cases)) + ' cases'

        with tqdm(total=max_count) as pbar:
            for index, case in enumerate(cases):

                pbar.update(1)
                print(case)
                id = str(case)
                if os.path.isdir(Path.joinpath(path, case, 'lesion_masks/')): # check if case has mask and therefore hemorrhage
                    #try:
                    # if case has hemorrhage, extract metadata from dataframe
                    # TODO: fix insilicoICH metadata generation (currently messy with added strings and brackets)
                    temp_df = metadata_dropna.loc[metadata_dropna['Name'] == case]

                    attenuation = float(temp_df['LesionAttenuation(HU)'].unique()[0].replace('[','').replace(']',''))
                    volume = temp_df['LesionVolume(mL)'].apply(lambda x: x.replace('[','').replace(']','')).astype(float).sum()
                    lesion_type = temp_df['Subtype'].unique()[0].replace('[','').replace(']','')

                    attenuation_list.append(attenuation)
                    volume_list.append(volume)
                    type_list.append(lesion_type)
                    labels_list.append(1)

                else: # case has no mask, therefore no hemorrhage
                    attenuation_list.append('NaN')
                    volume_list.append('NaN')
                    type_list.append('None')
                    labels_list.append(0)

                dcm_path = Path.joinpath(path, case, 'dicoms/')

                img, files = prepare_images(dcm_path, options, id, save_dir)

                output = classify_images(img, options, model_path, device)
                preds_list.append(np.max(output[:, -1]))

                if options['verbose']: print(np.max(output[:, -1]))

        fpr_syn, tpr_syn, thresholds_syn = metrics.roc_curve(labels_list, preds_list, pos_label=1)
        roc_df_syn = pd.DataFrame(zip(fpr_syn, tpr_syn, thresholds_syn),columns = ["FPR","TPR","Threshold"])

        roc_auc_syn = metrics.auc(fpr_syn, tpr_syn)

        all_cases = cases.copy()

        dfsyn = pd.DataFrame()
        dfsyn['Name'] = all_cases
        dfsyn['TruthLabel'] = labels_list
        dfsyn['ModelPrediction'] = preds_list
        dfsyn['LesionAttenuation(HU)'] = attenuation_list
        dfsyn['LesionVolume(mL)'] = volume_list
        dfsyn['Subtype'] = type_list

        if options['save_csv']: dfsyn.to_csv(path / (str(dataset) + '_' + model + '_results.csv'))

        # another optional CSV - this is just the FPR, TPR, and thresholds that can easily be created again from *_results.csv
        # if options['save_csv']: roc_df_syn.to_csv(path / (str(dataset) + '_ROC.csv'))