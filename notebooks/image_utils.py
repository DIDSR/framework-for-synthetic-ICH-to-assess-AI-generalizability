import matplotlib.pyplot as plt
import numpy as np
import pydicom

from ipywidgets import interact, IntSlider
import os
import requests
import tarfile
import zipfile


def download_and_extract_archive(url, download_root, extract_root=None, filename=None, remove_finished=False):
    """Downloads an archive from a URL and extracts it.

    Args:
        url (str): URL of the archive to download.
        download_root (str): Directory to download the archive to.
        extract_root (str, optional): Directory to extract the archive to.
            If None, defaults to download_root.
        filename (str, optional): Name of the downloaded file. If None,
            the filename is inferred from the URL.
        remove_finished (bool, optional): If True, removes the downloaded
            archive after extraction.

    Returns:
        str: Path to the extracted directory.
    """

    if extract_root is None:
        extract_root = download_root

    if filename is None:
        filename = os.path.basename(url)
        if filename == "":  # Handle cases where basename is empty
            raise ValueError(f"Could not determine filename from URL: {url}")


    download_path = os.path.join(download_root, filename)

    if not os.path.exists(download_root):
        os.makedirs(download_root)

    # Download
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

    # Extract
    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_root)
        elif filename.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(download_path, "r") as tar_ref:
                tar_ref.extractall(extract_root)
        else:
            print(f"Unsupported archive type: {filename}")
            return None

    except (zipfile.BadZipFile, tarfile.ReadError) as e:
        print(f"Error extracting {download_path}: {e}")
        return None
    if remove_finished:
        os.remove(download_path)


def read_dicom(dcm_fname: str) -> np.ndarray:
    '''
    Reads dicom file and returns numpy array

    :param dcm_fname: dicom filename to be read
    '''
    dcm = pydicom.read_file(dcm_fname)
    return dcm.pixel_array + int(dcm.RescaleIntercept)


# https://radiopaedia.org/articles/windowing-ct?lang=us
display_settings = {
    'brain': (80, 40),
    'subdural': (300, 100),
    'stroke': (40, 40),
    'temporal bones': (2800, 600),
    'soft tissues': (400, 50),
    'lung': (1500, -600),
    'liver': (150, 30),
}


def ctshow(img, window='soft tissues', fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    # Define some specific window settings here
    if isinstance(window, str):
        if window not in display_settings:
            raise ValueError(f"{window} not in {display_settings}")
        ww, wl = display_settings[window]
    elif isinstance(window, tuple):
        ww = window[0]
        wl = window[1]
    else:
        ww = 6.0 * img.std()
        wl = img.mean()

    if img.ndim == 3:
        img = img[0].copy()

    ax.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)


def center_crop(img, thresh=-800, rows=True, cols=True):
    cropped = img[img.mean(axis=1)>thresh, :]
    cropped = cropped[:, img.mean(axis=0)>thresh]
    return cropped


def center_crop_like(img, ref, thresh=-800):
    cropped = img[ref.mean(axis=1)>thresh, :]
    cropped = cropped[:, ref.mean(axis=0)>thresh]
    return cropped


def scrollview(phantom, display='soft tissues'):
    interact(lambda idx: ctshow(phantom[idx], display),
             idx=IntSlider(value=phantom.shape[0]//2, min=0,
             max=phantom.shape[0]-1))


def load_vol(file_list):
    return np.stack(list(map(read_dicom, file_list)))


def get_lesion_coords(mask):
    z_loc = mask.mean(axis=1).mean(axis=1).argmax()
    x_loc = mask.mean(axis=0).mean(axis=0).argmax()
    y_loc = mask.mean(axis=0).mean(axis=1).argmax()
    return z_loc, x_loc, y_loc


def browse_studies(metadata, name='case_000', display='soft tissues',
                   slice_idx=0, f=None, ax=None):
    patient = metadata[(metadata['name']==name)].iloc[slice_idx]
    dcm_file = patient['image file']
    img = read_dicom(dcm_file)
    ww, wl = display_settings[display]
    minn = wl - ww/2
    maxx = wl + ww/2
    if (f is None) or (ax is None):
        f, ax = plt.subplots()
    im = ax.imshow(img, cmap='gray', vmin=minn, vmax=maxx)
    plt.colorbar(im, ax=ax, label=f'HU | {display} [ww: {ww}, wl: {wl}]')
    ax.set_title(patient['name'])


def study_viewer(metadata):
    viewer = lambda **kwargs: browse_studies(metadata, **kwargs)
    slices = list(range(168))  # fix later to be dynamic
    interact(viewer,
             name=metadata.name.unique(),
             display=display_settings.keys(),
             slice_idx=IntSlider(value=slices[len(slices)//2], min=min(slices), max=max(slices)))