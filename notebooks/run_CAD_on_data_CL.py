import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

from image_utils import *
from model_utils import *

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

#dataset_path = Path('../datasets')
dataset_path = Path('/projects01/didsr-aiml/jayse.weaver/insilicoich/')
#dataset_names = ['mixed_2500_count']
dataset_names = ['varymA_70_count500', 'varymA_280_count500', 'varymA_600_count500']

save_dir = '/home/jayse.weaver/temp_images/new/' # optional save directory if save_jpg or save_dcm = True

for dataset in dataset_names:
    print('Processing dataset: ' + str(dataset))
    path = dataset_path / dataset
    metadata = pd.concat([pd.read_csv(o) for o in path.rglob('metadata*.csv')], ignore_index=True)

    if options['save_csv']: metadata.to_csv(path / (str(dataset) + '_metadata.csv'))

    metadata_short = metadata.drop(metadata.columns[12:], axis=1)
    metadata_dropna = metadata_short.dropna()

    cases = sorted(os.listdir(path))
    cases = [case for case in cases if case.startswith('case')]

    labels_syn = []
    pred_syn = []
    type_syn = []
    volume_syn = []
    intensity_syn = []

    if options['native_resolution']:
        print('Processing in native resolution')
    else:
        print('Processing in 5 mm resolution')

    max_count = len(cases)
    description = 'Processing ' + str(len(cases)) + ' cases'

    with tqdm(total=max_count) as pbar:
        for index, case in enumerate(cases):

            print('processing : ' + str(case))

            pbar.update(1)

            id = str(case) + '_std'
            if os.path.isdir(Path.joinpath(path, case, 'lesion_masks/')): # check if case has mask and therefore hemorrhage
                labels_syn.append(1)

                # if case has hemorrhage, extract metadata from dataframe
                # TODO: fix insilicoICH metadata generation (currently messy with added strings and brackets)
                temp_df = metadata_dropna.loc[metadata_dropna['Name'] == case]

                lesion_type = temp_df['Subtype'].unique()[0] #.split("'")[1]
                type_syn.append(lesion_type)

                volume_syn.append(temp_df['LesionVolume(mL)'].sum())

                intensity = float(temp_df['LesionAttenuation(HU)'].unique()[0][1:-1])
                intensity_syn.append(intensity)

            else: # case has no mask, therefore no hemorrhage
                labels_syn.append(0)
                type_syn.append('None')
                volume_syn.append('NaN')
                intensity_syn.append('NaN')

            dcm_path = Path.joinpath(path, case, 'dicoms/')

            img, files = prepare_images(dcm_path, options, id, save_dir)

            output = classify_images(img, options, device)
            pred_syn.append(np.max(output[:, -1]))

            if options['verbose']: print(np.max(output[:, -1]))

    fpr_syn, tpr_syn, thresholds_syn = metrics.roc_curve(labels_syn, pred_syn, pos_label=1)
    roc_df_syn = pd.DataFrame(zip(fpr_syn, tpr_syn, thresholds_syn),columns = ["FPR","TPR","Threshold"])

    roc_auc_syn = metrics.auc(fpr_syn, tpr_syn)

    all_cases = cases.copy()

    dfsyn = pd.DataFrame()
    dfsyn['case'] = all_cases
    dfsyn['truth'] = labels_syn
    dfsyn['pred'] = pred_syn
    dfsyn['type'] = type_syn
    dfsyn['volume'] = volume_syn
    dfsyn['intensity'] = intensity_syn

    if options['save_csv']: dfsyn.to_csv(path / (str(dataset) + '_results.csv'))
    
    # another optional CSV - this is just the FPR, TPR, and thresholds that can easily be created again from *_results.csv
    # if options['save_csv']: roc_df_syn.to_csv(path / (str(dataset) + '_ROC.csv'))