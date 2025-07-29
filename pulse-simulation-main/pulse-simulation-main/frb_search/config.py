'''
This file is used to read the configuration file and store the values in a dictionary.
'''
import os
import glob
import configparser

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config_files = list(map(os.path.abspath, glob.glob(os.path.join(project_dir, 'config', '*.ini'))))
# if config_template.ini is in the files, put it at the beginning
config_files = sorted(config_files, key=lambda x: os.path.basename(x) != "config_template.ini")

config = configparser.ConfigParser()

# Read the config file
config.read(config_files)
