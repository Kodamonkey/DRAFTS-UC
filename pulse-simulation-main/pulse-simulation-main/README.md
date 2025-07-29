# Pulse SImulation And Search PIpeline. (PSIASPI) (Working title)
In this repository you will find `simulate_pulse.py` which is a basic script to simulate pulsar/FRB-like pulses. Currently there are 3 ways to model a pulse:
* Single gaussian in time
* 2D gaussian (time and frequency) (default)
* Frequency dependant model

Example of usage from terminal:
`simulate_pulse --bandpass 1800 1200 --fref 1800 --t_tot 1.5 --t_p 0.5 --f_m 1350 --fwhm_f 300 --fwhm_t 0.005 --channel_width 0.29296875 --time_resolution 0.001 
--dm 400 --subburst --ts --sed --noise --plot`

Resulting plot:

<img width="320" alt="readme_example_figure" src="https://github.com/astronomy-laboratory/pulse-simulation/assets/80717337/91571fd2-a31c-4427-a34d-df5561bddce5">


The `voltage_sim.py` script transfroms a synthetic pulse into it's voltage signal (the original pulse is recovered when performimg a PFB).

## Installation
To install the package clone or download the repository and then install it with pip from the root folder:
* `git clone https://github.com/astronomy-laboratory/pulse-simulation.git`
* `cd path/to/pulse-simulation`
* `pip install -e .`

## Execution
The scripts can be run from the terminal independently, either
* `python -m frb_search.<script name> --args`
* `<script name> --args`

## PRESTO Pipeline 
The scripts `frb_search.py`, `plot_waterfall.py` and `candidates.py` together integrate a search pipeline based on `PRESTO` software. The pipeline executes `rfifind`, 
`prepsubband` and `single_pulse_search.py` to obtain the pulse candidates and then filters duplicates to finally plot the candidates and save them into a pdf file for 
visual confirmation of the candidates. 

### Dependencies
* `numpy`
* `pandas`
* `matplotlib`
* [`sigpyproc3`](https://sigpyproc3.readthedocs.io/en/latest/install.html)
* [`your`](https://thepetabyteproject.github.io/your/0.6.6/)
* `astropy`
* `requests`
* `tqdm`
* `PyPDF2`
* `Tkinter`

Example of usage:
After having installed all necessary packages (you must install them one by one if you don't have them) proceed as follows:
In the `config` folder there is a template for the `config.ini` file you must create in the same folder. There, in the field `presto_image` you put the path to the `PRESTO` singularity image you have. If you want to use `Docker`, then in the field `presto_image_docker` put the name of the image. The `telegram_id` is your personal telegram id and the `telegram_token` must be requested through private communication. If you don't want to use the Telegram Bot simpy leave these fields empty.

If you want to run the pipeline inside a container, then you can build a `Docker` container with the `.dockerfile` inside the `config` folder. Run this command on the same directory in which you cloned the repo: `docker build -f /path/to/frb_search.dockerfile -t frb_search .` Then you can run the container with an interactive shell `docker run --rm -v ${PWD}:/root -w /root -it frb_search_v2`. You can use [`Quay`](https://quay.io/repository/singularity/docker2singularity) to convert the `Docker` image into a singularity image or request the singularity image file through private communication.

Finally, the main script from the pipeline `frb_search.py` can be run as follows (Use `frb_search -h` for complete list of arguments): 

`frb_search --filename 3098_0001_00_8bit.fil --dm_start 550 --num_dms 20 --dm_step 1 --downsample 8 --nsub 512 -m 0.04 --sigma 7 --dt 0.02 
--plot_window 0.5 --timeseries --spectral_average --save`

Example of plot of candidate:

![image](https://github.com/astronomy-laboratory/pulse-simulation/assets/80717337/a2fd1b07-b068-4868-8d9b-b04c5eafeda9)


