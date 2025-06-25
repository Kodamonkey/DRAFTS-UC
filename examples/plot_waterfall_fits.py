import argparse
from pathlib import Path

import numpy as np

from DRAFTS.io import get_obparams, load_fits_file
from DRAFTS.preprocessing import downsample_data
from DRAFTS.visualization import plot_waterfalls
from DRAFTS import config


def main(fits_path: Path, output_dir: Path, slice_len: int) -> None:
    # Extract observation parameters and load data
    get_obparams(str(fits_path))
    data = load_fits_file(str(fits_path))

    # Ensure two polarizations and flip for waterfall display
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    data = np.vstack([data, data[::-1, :]])

    # Down-sample using config parameters
    data = downsample_data(data)

    width_total = config.FILE_LENG // config.DOWN_TIME_RATE
    if width_total == 0:
        print("No data available for plotting")
        return

    slice_len = min(slice_len, width_total)
    time_slice = width_total // slice_len if width_total >= slice_len else 1

    plot_waterfalls(data, slice_len, time_slice, fits_path.stem, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot waterfall diagrams for a FITS file")
    parser.add_argument("fits_file", type=Path, help="Path to the FITS file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("waterfall_plots"),
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--slice-len",
        type=int,
        default=512,
        help="Number of time samples per waterfall image",
    )
    args = parser.parse_args()

    main(args.fits_file, args.output_dir, args.slice_len)
