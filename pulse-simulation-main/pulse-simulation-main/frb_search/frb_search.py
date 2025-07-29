# script to search for FRBs in data using the PRESTO pipeline

import os
import glob
import subprocess
import argparse
from astropy import units as u
import frb_search.plot_waterfall as pw
import frb_search.candidates as cands
from pathlib import Path
from .telegram_bot import send_message
from .config import config
import frb_search.converter as converter
import PyPDF2
from .search_gui import run_gui
import ast
# from memory_profiler import profile

def merge_pdfs(pdfs, output_file):
    pdf_writer = PyPDF2.PdfMerger()
    for pdf in pdfs:
        pdf_writer.append(pdf)
    with open(output_file, "wb") as out:
        pdf_writer.write(out)

def find_pdf_in_subdir(subdir):
    full_subdir_path = os.path.join(subdir, 'candidates')
    if not os.path.exists(full_subdir_path):
        return None
    for file in os.listdir(full_subdir_path):
        if file.endswith('.pdf'):
            return os.path.join(full_subdir_path, file)
    return None

def collect_pdfs(root_dir):
    pdf_files = []
    for subdir in os.listdir(root_dir):
        full_subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(full_subdir_path):
            pdf_file = find_pdf_in_subdir(full_subdir_path)
            if pdf_file:
                pdf_files.append(pdf_file)
    return pdf_files

def delete_pdfs(pdfs):
    for pdf_file in pdfs:
        os.remove(pdf_file)

def write_summary(summary_file, file_name, df, df_, num_plotted, downsample,
                  nsub, m, sigma, dt, plot_window, timeseries,
                  spectral_average, start_dm, num_dms, dm_step,
                  num_to_plot, stokes, skip_mask, convert_to_fil,
                  bandpass, time_resolution, nchannels,
                  flip_data, padding, remove_samples,
                  remove_fil, prepdata, wrap_pdfs, 
                  output_dir, logfiles_dir, rm_presto_output_files,
                  ncpus, rfifind_time, clip):
    with open(summary_file, 'w') as f:
        f.write('Summary of the FRB search pipeline\n')
        f.write('Filename: ' + file_name + '\n')
        f.write('Number of candidates after filtering: ' +
                str(len(df_)) + '\n')
        f.write('Number of candidates plotted: ' + str(num_plotted) + '\n')
        f.write('Total candidates obtained by PRESTO: ' + str(len(df)) + '\n')
        f.write('Downsampling factor: ' + str(downsample) + '\n')
        f.write('Number of subbands: ' + str(nsub) + '\n')
        f.write('Maximum width of pulse: ' + str(m) + '\n')
        f.write('Sigma threshold: ' + str(sigma) + '\n')
        f.write('Time in seconds to separate events: ' + str(dt) + '\n')
        f.write('Time in seconds to plot around each candidate: ' +
                str(plot_window) + '\n')
        f.write('Time series plotted: ' + str(timeseries) + '\n')
        f.write('Spectral average plotted: ' + str(spectral_average) + '\n')
        f.write('Start DM: ' + str(start_dm) + '\n')
        f.write('Number of DMs: ' + str(num_dms) + '\n')
        f.write('DM step size: ' + str(dm_step) + '\n')
        f.write('Number of candidates to plot: ' + str(num_to_plot) + '\n')
        f.write('Stokes parameter: ' + str(stokes) + '\n')
        f.write('Skip rfifind step: ' + str(skip_mask) + '\n')
        f.write('Converted to filterbank: ' + str(convert_to_fil) + '\n')
        f.write('Bandpass: ' + str(bandpass) + '\n')
        f.write('Time resolution: ' + str(time_resolution) + '\n')
        f.write('Number of frequency channels: ' + str(nchannels) + '\n')
        f.write('Flip data for conversion in frequency: ' + str(flip_data) + '\n')
        f.write('Padding added to the data in time samples: ' + str(padding) + '\n')
        f.write('Number of samples removed from the start of the data: ' + str(remove_samples) + '\n')
        f.write('Remove the .fil file after conversion and processing: ' + str(remove_fil) + '\n')
        f.write('Run PRESTO prepdata command instead of prepsubband: ' + str(prepdata) + '\n')
        f.write('Wrap pdfs into a single pdf file: ' + str(wrap_pdfs) + '\n')
        f.write('Output directory: ' + output_dir + '\n')
        f.write('Logfiles directory: ' + logfiles_dir + '\n')
        f.write('Remove PRESTO output files (rfifind, .inf, .dat, .singlepulse) after processing: ' + str(rm_presto_output_files) + '\n')
        f.write('Number of CPUs used for the PRESTO pipeline: ' + str(ncpus) + '\n')
        f.write('rfifind time in seconds: ' + str(rfifind_time) + '\n')
        f.write('clip threshold for dedispersion: ' + str(clip) + '\n')

    return

def create_versioned_directory(base_name):
    version = 1
    dir_name = base_name
    while os.path.exists(dir_name):
        version += 1
        dir_name = f"{base_name}_v{version}"
    os.makedirs(dir_name)
    return dir_name

def container_prefix(container_type, presto_image_path):
    if container_type == 'singularity':
        return f"singularity exec --bind $PWD {presto_image_path} "
    elif container_type == 'docker':
        cwd = os.getcwd()
        return f'docker run --rm -v {cwd}:/root -w /root {presto_image_path} '
    elif container_type == 'itself':
        return ""
    else:
        raise ValueError(f"Unknown container type: {container_type}")

def parser_args():
    '''
    Function to parse the command line arguments

    Input:
        None
    Output:
        args: object containing the command line arguments
    '''
    parser = argparse.ArgumentParser(
        description='Search for FRBs in data using the PRESTO pipeline'
    )
    parser.add_argument('--filename', nargs='+', type=str, default='all',
                        help='name of the .fil file')
    parser.add_argument('--gui', action='store_true',
                        help=('whether to use the GUI to input the search '
                              'parameters'))
    parser.add_argument('--dm_start', type=float, help='starting DM to search')
    parser.add_argument('--num_dms', type=int, help='number of DMs to search')
    parser.add_argument('--dm_step', type=float, help='DM step size')
    parser.add_argument('--downsample', type=int, default=1,
                        help='downsampling factor')
    parser.add_argument('--nsub', type=int, help='number of subbands')
    parser.add_argument('-m', type=float, help='max width pulse to search')
    parser.add_argument('--sigma', type=float,
                        help='sigma threshold to search for candidates')
    parser.add_argument('--dt', type=float,
                        help='time in seconds to separate events')
    parser.add_argument('--plot_window', type=float,
                        help='time in seconds to plot around each candidate')
    parser.add_argument('--timeseries', action='store_true',
                        help='whether to plot the time series')
    parser.add_argument('--spectral_average', action='store_true',
                        help='whether to plot the spectral average')
    parser.add_argument('--stokes', type=int, default=0,
                        help=('Stokes parameter to plot (0, 1, 2, 3) for '
                              '(I, Q, U, V)'))
    parser.add_argument('--num_to_plot', type=int, default=150,
                        help='number of candidates to plot')
    parser.add_argument('--mask', action='store_true',
                        help='if True then skip the rfifind step')
    parser.add_argument('--save', action='store_true',
                        help='whether to save the plots')
    parser.add_argument('--convert_to_fil', action='store_true',
                        help='convert ARTE logfiles to .fil files')
    parser.add_argument('--bandpass', type=float, nargs=2,
                        default=[1200, 1800],
                        help='bandpass of the data (low, high) in MHz')
    parser.add_argument('--time_resolution', type=float, default=0.01,
                        help='time resolution of the data in seconds')
    parser.add_argument('--nchannels', type=int, default=2048,
                        help='number of frequency channels in the data')
    parser.add_argument('--flip_data', action='store_true',
                        help='flip the data for conversion in frequency')
    parser.add_argument('--padding', type=int, default=0,
                        help='padding to add to the data in time samples')
    parser.add_argument('--remove_samples', type=int, default=0,
                        help=('number of samples to remove from the start '
                              'of the data '
                              'for conversion'))
    parser.add_argument('--remove_fil', action='store_true',
                        help=('remove the .fil file after conversion and '
                              'processing'))
    parser.add_argument('--prepdata', action='store_true',
                        help=('whether to run the PRESTO prepdata command '
                              'instead of prepsubband'))
    parser.add_argument('--wrap_pdfs', action='store_true',
                        help='whether to wrap the pdfs into a single pdf file')
    parser.add_argument('--output_dir', type=str, default=os.getcwd(),
                        help=('directory where to process the fils and '
                              'store output files'))
    parser.add_argument('--logfiles_dir', type=str, default=os.getcwd(),
                        help='directory where the logfiles are stored')
    parser.add_argument('--rm_presto_output_files', action='store_true',
                        help=('remove the PRESTO output files rfifind, '
                              ' .inf, .dat, .singlepulse after processing'))
    parser.add_argument('--ncpus', type=int, default=1,
                        help='number of CPUs to use for the PRESTO pipeline')
    parser.add_argument('--rfifind_time', type=float, default=2,
                        help=('time in seconds to use for the rfifind step, '
                              'default is 2 seconds'))
    parser.add_argument('--clip', type=float, default=6.0,
                        help=('clip threshold for dedispersion, the default '
                              'is 6.0, and for no clipping use 0.0'))
    parser.add_argument(
        '--container',
        choices=['singularity', 'docker', 'itself'],
        default='singularity',
        help=(
            'container type to use for the PRESTO pipeline, '
            'default is singularity, '
            'docker is also supported, '
            'itself is for running the pipeline inside a container'
        )
    )
    return parser.parse_args()

# @profile
def main():
    args = parser_args()
    # run the PRESTO pipeline
    # define the name of the .fil file
    gui = args.gui
    if gui:
        search_params = {}
        run_gui(search_params)
        if search_params:  # Check if search_params is not empty
            filename_input = search_params['filenames']
            if filename_input == 'all':
                filenames = 'all'
            else:
                filenames = list(ast.literal_eval(filename_input))
            start_dm = float(search_params['start_dm'])
            num_dms = int(search_params['num_dms'])
            dm_step = float(search_params['dm_step'])
            downsample = int(search_params['downsample'])
            nsub = int(search_params['nsub'])
            m = float(search_params['m'])
            sigma = float(search_params['sigma'])
            dt = float(search_params['dt'])
            plot_window = float(search_params['plot_window']) * u.s
            timeseries = search_params['timeseries']
            spectral_average = search_params['spectral_average']
            num_to_plot = int(search_params['num_to_plot'])
            stokes = int(search_params['stokes'])
            save = search_params['save']
            convert_to_fil = search_params['convert_to_fil']
            rm_presto_output_files = search_params['remove_presto']
            ncpus = int(search_params['ncpus'])
            wrap_pdfs = search_params['wrap_pdfs']
            remove_fil = search_params['remove_fil']
            rfifind_time = float(search_params['rfifind_time'])
        else:
            print('No search parameters provided, exiting...')
            return
    else:
        filenames = args.filename
        start_dm = args.dm_start
        num_dms = args.num_dms
        dm_step = args.dm_step
        downsample = args.downsample
        nsub = args.nsub
        m = args.m
        sigma = args.sigma
        dt = args.dt
        plot_window = args.plot_window * u.s
        timeseries = args.timeseries
        spectral_average = args.spectral_average
        num_to_plot = args.num_to_plot
        stokes = args.stokes
        save = args.save
        convert_to_fil = args.convert_to_fil
        ncpus = args.ncpus
        rm_presto_output_files = args.rm_presto_output_files
        wrap_pdfs = args.wrap_pdfs
        remove_fil = args.remove_fil
        rfifind_time = args.rfifind_time
        clip_presto = args.clip
        container = args.container

    from_candidate = 0
    rfifind_dir = 'mask'
    prepsubband_dir = 'timeseries_sps'

    if container == 'singularity':
        if (config.has_option('DEFAULT', 'presto_image') and
                config['DEFAULT']['presto_image'] and
                os.path.isfile(config['DEFAULT']['presto_image'])):
            presto_image_path = config['DEFAULT']['presto_image']
            prefix = container_prefix(container, presto_image_path)
        else:
            print('Please provide an existing path to the PRESTO singularity image')
            return
    elif container == 'docker':
        presto_image_path = config['DEFAULT']['presto_image_docker']
        if not presto_image_path:
            print('Please provide an existing path to the PRESTO docker image')
            return
        prefix = container_prefix(container, presto_image_path)
    elif container == 'itself':
        presto_image_path = ''
        prefix = container_prefix(container, presto_image_path)
    else:
        print('Unknown container type, exiting...')
        return

    # if no filename is provided, search for all .fil or .fits files
    # in the output directory
    if filenames == 'all' and not convert_to_fil:
        filenames = (
            glob.glob(args.output_dir + '/*.fil') +
            glob.glob(args.output_dir + '/*.fits')
        )

    if filenames == 'all' and convert_to_fil:
        # if no filename is provided, search for all files in the logfiles dir
        filenames = glob.glob(
            args.logfiles_dir + '/*'
        )

    if args.output_dir != os.getcwd():
        os.chdir(args.output_dir)

    if filenames == []:
        print(
            'No files were found in the directory, check you used all '
            'necessary args for your search\n'
        )
        print('Exiting...')
        return

    filenames.sort()

    for filename in filenames:
        filename_extension = Path(filename).suffix
        if filename_extension != '.fil' and filename_extension != '.fits' and not convert_to_fil:
            print('The file is not a filterbank nor a PSRFITS, please convert the file to a .fil or .fits file' + '\n')
            print('For ARTE logfiles you can convert them to filterbanks using the following flag: ' + '\n')
            print('--convert_to_fil' + '\n')
            print('And make sure to add the corresponding arguments for a succesful conversion, you can see them using the -h flag' + '\n')
            print('Exiting...')
            return
        if filename_extension != '.fil' and convert_to_fil:
            if gui:
                print('Parameters from GUI are not supported for conversion of ARTE logfiles, please use the command line arguments')
                return
            print('Converting ARTE logfile to filterbank...')
            bandpass = args.bandpass
            time_resolution = args.time_resolution
            nchannels = args.nchannels
            flip_data = args.flip_data
            padding = args.padding
            remove_samples = args.remove_samples
            logfiles_dir = Path(args.logfiles_dir)
            output_dir = Path(args.output_dir)
            full_path, start_time = converter.get_date_from_logfile(logfiles_dir, filename)
            frequencies = converter.get_frequencies(bandpass, nchannels)
            bandwidth = bandpass[0] - bandpass[1]
            path_save_fil = converter.get_output_path(output_dir, full_path)
            sigproc_object = converter.write_header_for_fil(
            path_save_fil.as_posix(),
            'ARTE',
            nchannels,
            bandwidth/nchannels,
            frequencies[0],
            time_resolution,
            start_time,
            32
        )    
            data, avg_pow,  clip, t, bases,flags, data_new,snr = converter.get_image_data_temperature([full_path.as_posix()])
            if remove_samples > 0:
                data = converter.remove_samples(data, remove_samples)

            if padding > 0:
                data = converter.add_padding(data, padding)

            if flip_data:
                data = converter.flip_data(data)

            data = data.astype('float' + str(32))

            converter.write_data_to_fil(sigproc_object, path_save_fil.as_posix(), data)
            print('Final shape:', data.shape) 
            print("File:", filename, "converted to filterbank!")
            filename = path_save_fil.name
            filename_extension = Path(filename).suffix
            # check if the conversion was succesful
            if not converter.check_conversion(path_save_fil):
                print('Conversion was not succesful, exiting...')
                return
            subprocess.run(["rm", 'readfile_output.txt'])
            print('Conversion was succesful, continuing with the FRB search pipeline...')

        filename_without_extension = Path(filename).stem
        print('Starting the FRB search pipeline, file is being processed.')
        print('Filename:', Path(filename).name + ' is being processed.')
        print('Data will be processed in the directory:', os.getcwd())

        start_message = 'Starting the FRB search pipeline, file is being processed\.'
        send_message(start_message)

        if args.mask:
            print('Skipping rfifind step...')
            print('Now starting prepsubband...')
        else:
            subprocess.run(f"{prefix}rfifind -time {rfifind_time} -o output -ncpus {ncpus} {filename}", shell=True)
            print('Done running rfifind!, now running prepsubband...')
            send_message('Done running rfifind\!, now running prepsubband')
        if args.prepdata:
            print('Running prepdata instead of prepsubband...')
            send_message('Running prepdata instead of prepsubband')
            dms = [start_dm + i*dm_step for i in range(num_dms)]
            for dm in dms:
                cmd = (f"{prefix}prepdata -nobary -dm {dm} -downsamp {downsample} "
                        f"-mask output_rfifind.mask -o prep_output_DM{dm} "
                        f"-ncpus {ncpus} -clip {clip_presto} {filename}")
                subprocess.run(cmd, shell=True)
            print('Done running prepdata!, now running single_pulse_search.py...')
            send_message('Done running prepdata\!, now running single\_pulse\_search\.py')
        else:
            # Run prepsubband
            cmd = (f"{prefix}prepsubband -nobary -lodm {start_dm} -dmstep {dm_step} -numdms {num_dms} "
                    f"-downsamp {downsample} -mask output_rfifind.mask -nsub {nsub} "
                    f"-runavg -o prep_output -ncpus {ncpus} {filename}")
            subprocess.run(cmd, shell=True)

            print('Done running prepsubband!, now running single_pulse_search.py...')

            send_message('Done running prepsubband\!, now running single\_pulse\_search\.py')
        # Run single_pulse_search.py
        subprocess.run(f"{prefix}single_pulse_search.py -m {m} -t {sigma} -b *.dat", shell=True)

        print('Done running single_pulse_search.py!, now reading in candidates...')

        send_message('Done running single\_pulse\_search\.py\!, now reading in candidates')

    # read in the candidates from the .singlepulse files
        output = 'candidates.csv'
        output_filtered = '{}_filtered.{}'.format(*output.split('.'))
        df, df_ = cands.candidates_file(dt)

        if len(df) == 0:
            print('No candidates found in ' + filename + ', exiting...')
            send_message('No candidates found, exiting')
            os.makedirs(rfifind_dir, exist_ok=True)
            subprocess.run(['mv', 'output_rfifind.mask', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.bytemask', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.inf', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.ps', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.rfi', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.stats', rfifind_dir])
            subprocess.run(["rm *.dat "], shell=True)
            subprocess.run(["rm *.inf "], shell=True)
            subprocess.run(["rm *.singlepulse "], shell=True)
            subprocess.run(["rm prep_output_singlepulse.ps "], shell=True)
            summary_file = filename_without_extension + '_summary_0candidates.txt'
            dir_name = create_versioned_directory(filename_without_extension)
            write_summary(summary_file, filename_without_extension, df, df_, 0, downsample, nsub, m, sigma,
                     dt, plot_window, timeseries, spectral_average, start_dm, num_dms, dm_step,
                        num_to_plot, stokes, args.mask, convert_to_fil, args.bandpass, args.time_resolution, args.nchannels,
                     args.flip_data, args.padding, args.remove_samples, remove_fil,
                     args.prepdata, wrap_pdfs, args.output_dir,
                     args.logfiles_dir, rm_presto_output_files, ncpus, rfifind_time, clip_presto)
            readme = 'readme.txt'
            with open(readme, 'w') as f:
                f.write('No candidates found in ' + filename + '\n')
                f.write('If you want to run the pipeline again, but want to keep the output_rfifind.mask file, ' +
                        'place it in the same directory as the .fil file and use the --mask flag or tick the box in the GUI'
                         + '\n')

            subprocess.run(["mv", readme, dir_name])
            subprocess.run(["mv", summary_file, dir_name])
            subprocess.run(["mv", rfifind_dir, dir_name])
            if args.remove_fil and convert_to_fil:
                subprocess.run(["rm", filename])
                print('Removed ' + filename)
            print('Summary file created: ' + summary_file)
            print('No candidates found in ' + filename)
            print('Continuing with the next file...')
            continue

        if save:
            df.to_csv(output, index=False)
            df_.to_csv(output_filtered, index=False)
        df_.reset_index(drop=True, inplace=True)

        if len(df_) > num_to_plot:
            num_plotted = num_to_plot
            print('Plotting ' + str(num_to_plot) + ' candidates out of ' + str(len(df)) + ' candidates...')

            send_message('Plotting ' + str(num_to_plot) + ' candidates out of ' + str(len(df)) + ' candidates')
        else:
            num_plotted = len(df_)
            print('Plotting ' + str(len(df_)) + ' candidates out of ' + str(len(df)) + ' candidates...')

            send_message('Plotting ' + str(len(df_)) + ' candidates out of ' + str(len(df)) + ' candidates')

        pw.plot_waterfall_from_df(Path(filename).name, df_, plot_window, downsample, filename_extension, num_to_plot, from_candidate,
                                  stokes=stokes, ts=timeseries, save=save, sed=spectral_average)
        print('Done plotting candidates!, now rearranging files...')

        send_message('Done plotting candidates\!, now rearranging files')
        # plot the candidates
        # move rfifind files to mask directory
        dir_name = create_versioned_directory(filename_without_extension)
        if rm_presto_output_files:
            subprocess.run(["rm *.dat "], shell=True)
            subprocess.run(["rm *.inf "], shell=True)
            subprocess.run(["rm *.singlepulse "], shell=True)
            subprocess.run(['mv', 'prep_output_singlepulse.ps', dir_name])
            os.remove('output_rfifind.mask')
            os.remove('output_rfifind.bytemask',)
            os.remove('output_rfifind.rfi')
            os.remove('output_rfifind.stats')
            subprocess.run(['mv', 'output_rfifind.ps', dir_name])
            subprocess.run(['mv', 'candidates.csv', dir_name])
            subprocess.run(['mv', 'candidates_filtered.csv', dir_name])
            subprocess.run(['mv *.pdf ' + dir_name], shell=True)
        else:
            os.makedirs(rfifind_dir, exist_ok=True)
            os.makedirs(prepsubband_dir, exist_ok=True)
            os.makedirs('candidates', exist_ok=True)
        # move rfifind files to mask directory
            subprocess.run(['mv', 'output_rfifind.mask', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.bytemask', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.inf', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.ps', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.rfi', rfifind_dir])
            subprocess.run(['mv', 'output_rfifind.stats', rfifind_dir])

            # move prepsubband files to timeseries_sps directory
            subprocess.run(["mv *.dat " + prepsubband_dir], shell=True)
            subprocess.run(["mv *.inf " + prepsubband_dir], shell=True)
            subprocess.run(["mv *.singlepulse " + prepsubband_dir], shell=True)
            subprocess.run(["mv", "prep_output_singlepulse.ps", prepsubband_dir])
            # move candidates tables and pdf to candidates directory
            subprocess.run(["mv *.csv" + " candidates"], shell=True)
            subprocess.run(["mv *.pdf" + " candidates"], shell=True)

        # move all files to a directory with the name of the .fil file
            subprocess.run(["mv", rfifind_dir, dir_name])
            subprocess.run(["mv", prepsubband_dir, dir_name])
            subprocess.run(["mv", "candidates", dir_name])

        # make summary file
        summary_file = dir_name + '_summary.txt'
        write_summary(summary_file, filename_without_extension, df, df_, num_plotted, downsample, nsub, m, sigma,
                     dt, plot_window, timeseries, spectral_average, start_dm, num_dms, dm_step,
                        num_to_plot, stokes, args.mask, convert_to_fil,
                     args.bandpass, args.time_resolution, args.nchannels,
                     args.flip_data, args.padding, args.remove_samples, remove_fil,
                     args.prepdata, wrap_pdfs, args.output_dir,
                     args.logfiles_dir, rm_presto_output_files, ncpus, rfifind_time, clip_presto)

        subprocess.run(["mv", summary_file, dir_name])
        print('Summary file created: ' + summary_file)
        if remove_fil and convert_to_fil:
            subprocess.run(["rm", filename])
            print('Removed ' + filename)

        print(Path(filename).name + ' Done!, Hope you found some FRBs/Pulsars!')
        end_message = 'Done\!, Hope you found some FRBs/Pulsars\!'
        send_message(end_message)

    if wrap_pdfs:
            pdf_files = collect_pdfs(os.getcwd())
            merge_pdfs(pdf_files, 'all_candidates.pdf')
            delete_pdfs(pdf_files)
            print('All pdfs wrapped into all_candidates.pdf')

if __name__ == '__main__':
    main()
