import numpy as np
import pandas as pd
import os
import glob
import argparse
from pathlib import Path
import ast

def get_candidates(filename):
    '''
    Function to read in the candidates from the .singlepulse file
    and return a pandas dataframe
    
    Input:
        filename: name of the .singlepulse file
    Output:
        df: pandas dataframe containing the candidates
    '''
    # read in the file
    try: 
        df = pd.read_csv(filename, delimiter=r'\s+', skiprows=1, 
                         names=['DM', 'Sigma', 'Time(s)', 'Sample', 'Downfact'])
        return df
    except:
        return pd.DataFrame()

def concat_candidates(filenames):
    '''
    Function to concatenate the candidates from multiple .singlepulse files
    and return a pandas dataframe
    
    Input:
        filenames: list of .singlepulse files
    Output:
        df: pandas dataframe containing the candidates
    '''
    # read in the first file
    df = get_candidates(filenames[0])
    # loop through the remaining files
    for filename in filenames[1:]:
        # read in the file
        df_temp = get_candidates(filename)
        # concatenate the dataframes
        df = pd.concat([df, df_temp], ignore_index=True)
    return df

def separate_events(df, dt):
    '''
    Function to separate events that are dt seconds apart to distinguish
    different times for the same event
    
    Input:
        df: pandas dataframe containing the candidates
        dt: time in seconds to separate events
    Output:
        df: pandas dataframe containing the final candidates
    '''
    # Sort the DataFrame by time and sigma in descending order
    df_sorted = df.sort_values(by=['Time(s)', 'Sigma'], ascending=[True, False])

    # Create a list to store the selected rows
    selected_rows = []

    # Iterate through the sorted DataFrame
    for index, row in df_sorted.iterrows():
        time = row['Time(s)']
        sigma = row['Sigma']

        # Check if there are any selected rows that are within the time threshold
        similar_rows = [selected_row for selected_row in selected_rows if abs(selected_row['Time(s)'] - time) <= dt]

        if not similar_rows:
            selected_rows.append(row)
        else:
            # Compare sigma values and update if the current row has a higher sigma
            max_sigma = max(similar_rows, key=lambda x: x['Sigma'])['Sigma']
            if sigma > max_sigma:
                # Replace the existing row with the current row
                selected_rows = [selected_row for selected_row in selected_rows if abs(selected_row['Time(s)'] - time) > dt]
                selected_rows.append(row)

    # Create a new DataFrame from the selected rows
    selected_df = pd.DataFrame(selected_rows)
    
    # Sort the selected DataFrame by sigma in descending order
    selected_df = selected_df.sort_values(by=['Sigma'], ascending=[False])

    return selected_df

def candidates_file(dt, file_extension='singlepulse'):
    '''
    Function to read in the candidates from multiple files
    and return a pandas dataframe

    Input:
        dt: time in seconds to separate events
        file_extension: extension of the singlepulse files
    Output:
        df: pandas dataframe containing the candidates
    '''
    current_dir = os.getcwd()
    files = glob.glob(current_dir + '/*.{}'.format(file_extension))
    #files = glob.glob(os.path.join(directory, '*.{}'.format(file_extension)))
    dfs = []
    for file in files:
        dfs.append(get_candidates(file))

    # concatenate the dataframes
    df = pd.concat(dfs, ignore_index=True)
    # separate events that are dt seconds apart
    if len(df.index)<1:
        df_ = df
    else:
        df_ = separate_events(df, dt)

    return df, df_


def find_matching_candidates(df, df_, dt):
    '''
    Function to find the number of true positive, false positive, and false negative
    candidates between the original pulses and the candidates obtained by PRESTO when
    performing an injection test

    Input:
        df: pandas dataframe containing the original pulses
        df_: pandas dataframe containing the candidates obtained by PRESTO
        dt: time in seconds to separate events
    Output:
        tp: number of true positive candidates
        fp: number of false positive candidates
        fn: number of false negative candidates
    '''
    true_positives = 0
    false_positives = 0
    df = df.rename(columns={'Arrival': 'Time(s)'})
    for _, candidate in df_.iterrows():
        time_difference = abs(df['Time(s)'] - candidate['Time(s)'])
        is_match = any(time_difference <= dt)
        
        if is_match:
            true_positives += 1
        else:
            false_positives += 1
    
    return true_positives, false_positives
    

def parser():
    '''
    Function to parse the command line arguments
    '''
    parser = argparse.ArgumentParser(description='Merge candidates from multiple singlepulse files')
    parser.add_argument('-dt', '--dt', help='Time in seconds to separate events', type=float, required=True)
    parser.add_argument('-o', '--output', help='Name of the output file', type=str)
    parser.add_argument('-s', '--save', help='Save the datframe', action='store_true')
    parser.add_argument('-injection_stats', '--injection_stats', 
                        help='Calculate the number of true positive, false positive', action='store_true')
    parser.add_argument('-in', '--injection', help='Name of the file with injected info', type=str)
    parser.add_argument('-p', '--presto_file', help='Name of the file with the candidates obtained by PRESTO', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parser()
    dt = args.dt
    save = args.save
    injection_stats = args.injection_stats
    if save:
        output = args.output
        output_filtered = '{}_filtered.{}'.format(*output.split('.'))
        df, df_ = candidates_file(dt)
        df.to_csv(output, index=False)
        df_.to_csv(output_filtered, index=False)
    elif injection_stats:
        injection_file = Path(args.injection)
        presto_file = Path(args.presto_file)
        with open(injection_file, 'r') as file:
            contents = file.read()
            dictionary = ast.literal_eval(contents)
            dfs = []
            for i in range(len(dictionary)):
                df_c = pd.DataFrame.from_dict(dictionary[i], orient='index')
                df_c = df_c.transpose()
                dfs.append(df_c)
            df_injected = pd.concat(dfs, ignore_index=True)
            presto_df = pd.read_csv(presto_file)
            tp, fp = find_matching_candidates(df_injected, presto_df, dt)

            print('True Positives: {}'.format(tp))
            print('False Positives: {}'.format(fp))


if __name__ == '__main__':
    main()