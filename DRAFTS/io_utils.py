import numpy as np
from ..io.io import load_fits_file
from ..io.filterbank_io import load_fil_file
from ..preprocessing.preprocessing import downsample_data

def load_and_preprocess_data(fits_path):
    """Carga y preprocesa los datos del archivo FITS o FIL."""
    if fits_path.suffix.lower() == ".fits":
        data = load_fits_file(str(fits_path))
    else:
        data = load_fil_file(str(fits_path))
    data = np.vstack([data, data[::-1, :]])
    return downsample_data(data)
