import os, re, sys
import numpy as np
import pandas as pd
from astropy.io import fits

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('default')
sns.set_color_codes()

import torch, torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from BinaryClass.binary_model import SPPResNet, BinaryNet


### 读取fits文件，只保留两维数据
def load_fits_file(file_name, reverse_flag=False):
    """Load a FITS file and return the data array in shape (time, pol, channel)."""
    data_array = None
    
    try:
        with fits.open(file_name, memmap=True) as hdul:
            # Check for SUBINT HDU with DATA column
            if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                subint = hdul["SUBINT"]
                hdr = subint.header
                data_array = subint.data["DATA"]
                nsubint = hdr["NAXIS2"]
                nchan = hdr["NCHAN"]
                npol = hdr["NPOL"]
                nsblk = hdr["NSBLK"]
                data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
                data_array = data_array.reshape(nsubint * nsblk, npol, nchan)
                data_array = data_array[:, :2, :]
            else:
                # Fallback to fitsio
                import fitsio
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    data_array = temp_data["DATA"].reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])[:, :2, :]
                else:
                    total_samples = h.get("NAXIS2", 1) * h.get("NSBLK", 1)
                    num_pols = h.get("NPOL", 2)
                    num_chans = h.get("NCHAN", 512)
                    data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
    except Exception as e:
        print(f"[Error loading FITS with astropy/fitsio] {e}")
        try:
            # Final fallback with astropy
            with fits.open(file_name) as f:
                data_hdu = None
                for hdu_item in f:
                    if hdu_item.data is not None and isinstance(hdu_item.data, np.ndarray) and hdu_item.data.ndim >= 3:
                        data_hdu = hdu_item
                        break
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                    
                h = data_hdu.header
                raw_data = data_hdu.data
                data_array = raw_data.reshape(
                    h["NAXIS2"] * h.get("NSBLK", 1), 
                    h.get("NPOL", 2), 
                    h.get("NCHAN", raw_data.shape[-1])
                )[:, :2, :]
        except Exception as e_astropy:
            print(f"Final failure loading with astropy: {e_astropy}")
            raise
    
    if data_array is None:
        raise ValueError(f"Could not load data from {file_name}")

    if reverse_flag:
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
    return data_array


def preprocess_data(data, exp_cut=5):

    data         = data.copy()
    data         = data + 1
    w, h         = data.shape
    data        /= np.mean(data, axis=0)
    vmin, vmax   = np.nanpercentile(data, [exp_cut, 100-exp_cut])
    data         = np.clip(data, vmin, vmax)
    data         = (data - data.min()) / (data.max() - data.min())

    return data

def plot_burst(data, filename, block):

    fig          = plt.figure(figsize=(5, 5))
    gs           = gridspec.GridSpec(4, 1)

    w, h         = data.shape
    profile      = np.mean(data, axis=1)
    time_start   = ((fits_number - 1) * file_leng + block * block_size) * time_reso
    peak_time    = time_start + np.argmax(profile) * time_reso

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(gs[0, 0])
    plt.plot(profile, color='royalblue', alpha=0.8, lw=1)
    plt.scatter(np.argmax(profile), np.max(profile), color='red', s=100, marker='x')
    plt.xlim(0, w)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(gs[1:, 0])
    plt.imshow(data.T, origin='lower', cmap='mako', aspect='auto')
    plt.scatter(np.argmax(profile), 0, color='red', s=100, marker='x')
    plt.yticks(np.linspace(0, h, 6), np.int64(np.linspace(freq.min(), freq.max(), 6)))
    plt.xticks(np.linspace(0, w, 6), np.round(time_start + np.arange(6) * time_reso * block_size / 5, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.savefig('{}{}-{:0>4d}-{}.jpg'.format(save_path, filename, block, peak_time), format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

    return None


if __name__ == '__main__':

    ### path config
    down_sampling_rate        = 8
    DM                        = 527

    prob                      = 0.6
    block_size                = 512
    base_model                = 'resnet18'
    model_path                = './models/class_resnet18.pth'

    date_path                 = './Data/'
    save_path                 = './Results/Classification/'
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    file_list                 = np.sort(os.listdir(date_path))
    file_list                 = np.append(file_list, file_list[-1])

    ### file params read
    with fits.open(date_path + file_list[0]) as f:
        time_reso             = f[1].header['TBIN'] * down_sampling_rate
        freq_reso             = f[1].header['NCHAN']
        file_leng             = f[1].header['NAXIS2'] * f[1].header['NSBLK']  // down_sampling_rate
        freq                  = f[1].data['DAT_FREQ'][0, :].astype(np.float64)

    reverse_flag              = False
    if freq[0] > freq[-1]:
        reverse_flag          = True
        freq                  = np.array(freq[::-1])

    # Filter out zero frequencies to avoid divide by zero
    freq_nonzero              = freq[freq > 0]
    if len(freq_nonzero) == 0:
        print("Error: All frequencies are zero or invalid")
        sys.exit(1)
    
    # Use only non-zero frequencies for calculations
    freq                      = freq_nonzero
    freq_reso                 = len(freq)

    ### time delay correct
    dds                       = (4.15 * DM * (freq**-2 - freq.max()**-2) * 1e3 / time_reso).astype(np.int64)
    
    # Handle potential overflow in dds
    dds                       = np.clip(dds, 0, file_leng * 10)  # Reasonable upper bound
    
    if file_leng % 512:
        redundancy            = ((file_leng // 512) + 1) * 512 - file_leng
    else:
        redundancy            = 0

    comb_leng                 = int(dds.max() / file_leng) + 1
    comb_file_leng            = (file_leng + redundancy + dds.max()) * down_sampling_rate
    down_file_leng            = file_leng + redundancy

    # Add safety check for overflow
    if comb_file_leng < 0 or comb_file_leng > 1e9:  # Reasonable upper bound
        print(f"Warning: comb_file_leng seems too large: {comb_file_leng}")
        comb_file_leng        = file_leng * down_sampling_rate * 2  # Fallback value

    ### model config
    model                     = BinaryNet(base_model, num_classes=2).to(device)
#    model                     = SPPResNet(base_model, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    ### read data
    for i in range(len(file_list) - 1):
        raw_data              = load_fits_file(date_path + file_list[i], reverse_flag)
        fits_number           = i + 1
        filename              = file_list[i].split('.fits')[0]
        print(filename)

        for j in range(comb_leng):
            if i + j + 1      < len(file_list):
                raw_data      = np.append(raw_data, load_fits_file(date_path + file_list[i+j+1], reverse_flag), axis=0)
        if raw_data.shape[0]  < comb_file_leng:
            raw_data          = np.append(raw_data, np.random.rand(comb_file_leng-raw_data.shape[0], 2, freq_reso) * raw_data.max() / 2, axis=0)

        raw_data              = raw_data[:comb_file_leng, :, :]
        
        # Check if reshape is possible before attempting
        expected_size         = (raw_data.shape[0] // down_sampling_rate) * down_sampling_rate * 2 * freq_reso
        actual_size           = raw_data.size
        
        if expected_size != actual_size:
            print(f"Warning: Size mismatch. Expected: {expected_size}, Actual: {actual_size}")
            # Adjust to make reshape possible
            target_length     = (raw_data.shape[0] // down_sampling_rate) * down_sampling_rate
            raw_data          = raw_data[:target_length, :, :]
        
        data                  = np.mean(raw_data.reshape(raw_data.shape[0] // down_sampling_rate, down_sampling_rate, 2, freq_reso), axis=(1, 2)).astype(np.float32)

        new_data              = np.zeros((down_file_leng, freq_reso))
        for j in range(freq_reso):
            new_data[:, j]    = data[dds[j]: dds[j]+down_file_leng, j]
        data                  = np.mean(new_data.reshape(down_file_leng//512, 512, 512, freq_reso//512), axis=3)

        ### predict
        for j in range(data.shape[0]):
            data[j, :, :]     = preprocess_data(data[j, :, :])
        inputs                = torch.from_numpy(data[:, np.newaxis, :, :]).float().to(device)
        predict_res           = model(inputs)

        ### plot
        with torch.no_grad():
            predict_res       = predict_res.softmax(dim=1)[:, 1].cpu().numpy()
        blocks                = np.where(predict_res >= prob)[0]
        for block in blocks:
            plotres           = plot_burst(data[block], filename, block)
            np.save('{}{}-{:0>4d}.npy'.format(save_path, filename, block), data[block])