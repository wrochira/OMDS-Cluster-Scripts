"""
Notes:
* Before running this script on Viking, be sure to load an appropriate Python 3 module,
  e.g. with the command `module load lang/SciPy-bundle/2020.03-foss-2019b-Python-3.7.4`
"""

import os
import sys
import math
import time
import shutil
import statistics
import xml.etree.ElementTree

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as scistats
matplotlib.rcParams.update({'errorbar.capsize': 5})


# Globals
### Constants
OBS_TIMEPOINTS = [ 0, 0.03333, 0.06666, 0.09999, 0.13332, 0.16665, 0.19998, 0.23331, 0.26664, 0.29997, 0.3333, 0.36663, 0.39996, 0.43329, 0.46662, 0.49995, 0.53328, 0.56661, 0.59994, 0.63327, 0.6666, 0.69993, 0.73326, 0.76659, 0.79992, 0.83325, 0.86658, 0.89991, 0.93324, 0.96657, 0.9999 ]
OBS_DATAPOINTS = [ 0, 0.0029767, 0.0050122, 0.0072264, 0.0086977, 0.009889, 0.010522, 0.010981, 0.011506, 0.012154, 0.012248, 0.012361, 0.012455, 0.012771, 0.012979, 0.013139, 0.013295, 0.013527, 0.013463, 0.013404, 0.013382, 0.013477, 0.013626, 0.013696, 0.013713, 0.01374, 0.013832, 0.013819, 0.013804, 0.013799, 0.013784 ]
TIMEPOINTS = [ (x+1)/1000 for x in range(999) ]
RIP_DIR = './runs_in_progress/'
RS_DIR = './runs_saved/'
SIM_OUT_DIR_NAME = 'output_files'
RESULTS_DIR_NAME = 'RESULTS'
INDIVIDUAL_MSDS_DIR_NAME = 'Individual MSDs'
### Variables
RUN_NAME = ''
CURRENT_DIR = ''
SHOULD_MOVE = True
RESULTS = { }


def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
    print()


def least_squares_score(timepoints, msds):
    point_deltas = [ ]
    for x, true_msd in zip(timepoints, msds):
        ideal_msd = -0.1325*x**6 + 0.5272*x**5 - 0.8634*x**4 + 0.7492*x**3 - 0.3695*x**2 + 0.1029*x - 9E-05 # Function to approximate experimental data
        point_deltas.append(true_msd - ideal_msd)

    score = math.sqrt(sum([ x**2 for x in point_deltas ]))
    #score = 100 * max(0, (0.4 - score) / 0.4)

    return score


def setup():
    global RUN_NAME
    global CURRENT_DIR

    clear_screen()
    runs = sorted(next(os.walk(RIP_DIR))[1])
    print('Available runs:')
    for i, run in enumerate(runs):
        print(str(i+1) + ')', run)
    print()
    chosen_dir = ''
    while not os.path.isdir(chosen_dir):
        try:
            choice = int(input('> ').strip())
            chosen_dir = os.path.join(RIP_DIR, runs[choice-1])
        except:
            chosen_dir = ''
    RUN_NAME = runs[choice-1]
    CURRENT_DIR = chosen_dir

    if not os.path.isdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME)):
        os.mkdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME))


def perform_checks():
    global SHOULD_MOVE
    print()
    print('Move run directory when finished?')
    SHOULD_MOVE = input('> ').strip().upper() in [ 'Y', 'YES' ]


def load_and_calculate_results():
    global RESULTS

    # Check number of parameter sets
    try:
        paramsets_file_fn = next(f for f in os.listdir(CURRENT_DIR) if f[-4:] == '.xml')
    except:
        print('Error finding parameter sets file')
        exit(1)
    e = xml.etree.ElementTree.parse(os.path.join(CURRENT_DIR, paramsets_file_fn)).getroot()
    num_paramsets = len(e.findall('PARAMETERS'))

    # Get relevant output files
    all_output_files = sorted(os.listdir(os.path.join(CURRENT_DIR, SIM_OUT_DIR_NAME)))
    xml_output_files = [ x for x in all_output_files if x[-4:] == '.xml' ]

    incomplete_file_names = set()

    for paramset_id in range(num_paramsets):
        paramset_results = { 'n' : 0,
                             'result_msds' : { },
                             'result_stds' : { },
                             'timing_mean' : None,
                             'timing_std' : None,
                             'lss' : None }
        individual_file_results = [ ]

        # Get data from all available output files for each parameter set
        paramset_output_files = [ x for x in xml_output_files if ('paramset-' + str(paramset_id) + '_') in x ]
        for filename in paramset_output_files:
            filepath = os.path.join(CURRENT_DIR, SIM_OUT_DIR_NAME, filename)
            e = xml.etree.ElementTree.parse(filepath).getroot()

            file_results = { 'n' : 0,
                             'result_msds' : { },
                             'result_stds' : { },
                             'timing_mean' : None,
                             'timing_std' : None }

            # Load results
            try:
                for res in e.find('RESULTS').findall('RES'):
                    timepoint = float(res.get('T'))
                    msd = float(res.get('MSD'))
                    std = float(res.get('STD'))
                    num = int(float(res.get('NUM')))
                    file_results['n'] = num
                    file_results['result_msds'][timepoint] = msd
                    file_results['result_stds'][timepoint] = std
            except:
                incomplete_file_names.add(filename)
                continue

            # Load time stats
            try:
                stats = e.find('STATISTICS').findall('STAT')
                for stat in stats:
                    stat_name = next(value for key, value in stat.items() if key == 'NAME')
                    stat_value = next(value for key, value in stat.items() if key == 'VALUE')
                    if stat_name == 'MEAN_SIMULATION_DURATION':
                        file_results['timing_mean'] = float(stat_value)
                    elif stat_name == 'STD_SIMULATION_DURATION':
                        file_results['timing_std'] = float(stat_value)
            except:
                incomplete_file_names.add(filename)
                continue

            individual_file_results.append(file_results)

        # Count total number of runs
        for file_results in individual_file_results:
            paramset_results['n'] += file_results['n']

        # Calculate weighted average of timings
        timing_mean = 0
        timing_std = 0
        for file_results in individual_file_results:
            timing_mean += file_results['n'] * file_results['timing_mean']
            timing_std += file_results['n'] * file_results['timing_std']**2
        timing_mean /= paramset_results['n']
        timing_std = math.sqrt(timing_std / paramset_results['n'])
        paramset_results['timing_mean'] = timing_mean
        paramset_results['timing_std'] = timing_std

        # Calculate weighted average of results
        for timepoint in TIMEPOINTS:
            results_msd = 0
            results_std = 0
            for file_results in individual_file_results:
                results_msd += file_results['n'] * file_results['result_msds'][timepoint]
                results_std += file_results['n'] * file_results['result_stds'][timepoint]**2
            results_msd /= paramset_results['n']
            results_std = math.sqrt(results_std / paramset_results['n'])
            paramset_results['result_msds'][timepoint] = results_msd
            paramset_results['result_stds'][timepoint] = results_std

        # Calculate LSS
        paramset_results['lss'] = least_squares_score(list(paramset_results['result_msds'].keys()),
                                                      list(paramset_results['result_msds'].values()))

        RESULTS[paramset_id] = paramset_results

    clear_screen()
    if len(incomplete_file_names) > 0:
        print('WARNING: Some instances did not complete enough runs to be included in the analyses:')
        for filename in sorted(incomplete_file_names):
            print('*', filename)
    print()
    print('The run counts for each parameter set are:')
    for paramset_id, paramset_results in RESULTS.items():
        print(paramset_id, paramset_results['n'])
    print()
    input('> ')


def write_results():
    paramset_ids = list(RESULTS.keys())
    result_msds = [ list(x['result_msds'].values()) for x in RESULTS.values() ]
    result_stds = [ list(x['result_stds'].values()) for x in RESULTS.values() ]
    runtime_means = [ x['timing_mean'] for x in RESULTS.values() ]
    runtime_stds = [ x['timing_std'] for x in RESULTS.values() ]
    least_squares_scores = [ x['lss'] for x in RESULTS.values() ]

    # Write run times to CSV
    with open(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Run Times.csv'), 'w') as outfile:
        outfile.write('Parameter Set,Run Time Mean (s),Run Time S.D. (s)\n')
        for paramset_id, paramset_results in RESULTS.items():
            outfile.write(str(paramset_id) + ',' + str(round(paramset_results['timing_mean'], 9)) + ',' + str(round(paramset_results['timing_std'], 9)) + '\n')

    # Write MSDs to CSV
    with open(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'MSDs.csv'), 'w') as outfile:
        outfile.write('Time (s),' + ','.join([ 'Set ' + str(x) + ',Set ' + str(x) for x in paramset_ids ]) + '\n')
        outfile.write(',' + ','.join(['MSD (µm^2),MSD S.D. (µm^2)'] * len(RESULTS.keys())) + '\n')
        for timepoint in TIMEPOINTS:
            outfile.write(str(timepoint) + ',' + ','.join([ str(round(paramset_results['result_msds'][timepoint], 9)) + ',' + str(round(paramset_results['result_stds'][timepoint], 9)) for paramset_results in RESULTS.values() ]) + '\n')

    # Write least squares scores to CSV
    with open(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Scores.csv'), 'w') as outfile:
        outfile.write('Parameter Set,Least Squares Score)\n')
        for paramset_id, paramset_results in RESULTS.items():
            outfile.write(str(paramset_id) + ',' + str(round(paramset_results['lss'], 2)) + '\n')

    # Plot bar chart of least squares scores
    ax = plt.subplot(111)
    ax.bar(paramset_ids, least_squares_scores, width=0.5, align='center')
    #ax.set_ylim(0, 100)
    plt.xticks(paramset_ids, [ str(x) for x in paramset_ids ])
    plt.xlabel('Parameter Set')
    plt.ylabel('Least Squares Score')
    plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Scores Graph.png'), dpi=600)
    plt.close()

    # Get graph specifics
    axis_label = ''
    axis_values = [ ]
    axis_value_type = ''
    do_individual_graphs = False
    do_linreg = False

    # Load discrepancies from variables.txt
    discrepancies = [ ]
    with open(os.path.join(CURRENT_DIR, 'variables.txt'), 'r') as infile:
        for line in infile.readlines():
            line = line.strip()
            if line:
                paramname = line.split(': ')[0]
                paramvals = line.split(': ')[1].split(', ')
                discrepancies.append((paramname, paramvals))

    # Prompt for graph specifics
    clear_screen()
    if len(discrepancies) > 0:
        print('Discrepancies:')
        for i, mm in enumerate(discrepancies):
            if mm[1].count(mm[1][0]) == len(mm[1]):
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], 'All ' + str(mm[1][0]))
            else:
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], ', '.join(mm[1]))
    else:
        print('No discrepancies.')
    print()
    print('Enter parameter set title')
    axis_label = input('> ').strip()
    print()
    while True:
        print('Enter parameter set values (comma seperated, length ' + str(len(paramset_ids)) + ')')
        axis_labels_text = input('> ').strip()
        axis_values = [ x.strip() for x in axis_labels_text.split(',') ]
        if len(axis_values) == len(paramset_ids):
            break
        print('Wrong length.')
    try:
        [ float(x) for x in axis_values ]
        print()
        print('Use proportional axis for runtimes?')
        if input('> ').strip().upper() in [ 'Y', 'YES' ]:
            axis_values = [ float(x) for x in axis_values ]
            axis_value_type = 'Number'
        else:
            axis_value_type = 'String'
    except:
        print('Axis labels will be strings.')
        axis_value_type = 'String'
    if axis_value_type == 'Number':
        print()
        print('Perform linear regression on runtimes?')
        do_linreg = input('> ').strip().upper() in [ 'Y', 'YES' ]
    print()
    print('Produce individual graphs for each parameter set?')
    do_individual_graphs = input('> ').strip().upper() in [ 'Y', 'YES' ]
    print()
    print('Finalising results...')
    print()

    # Linear regression fit for run times
    if do_linreg:
        slope, intercept, r_value, p_value, std_err = scistats.linregress(axis_values, runtime_means)

    # Plot scatter graph of run times with error bars, with linear regression fit on top
    ax = plt.subplot(111)
    ax.errorbar(axis_values, runtime_means, yerr=runtime_stds, fmt='o', linestyle='None')
    if do_linreg:
        xs = np.array(axis_values)
        ys = slope * xs + intercept
        ax.plot(xs, ys, '-r', color='black', label='y = ' + str(round(slope, 1)) + 'x + ' + str(round(intercept, 1)) + ', R2 = ' + str(round(r_value**2, 3)))
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
        leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
    plt.xlabel(axis_label)
    plt.ylabel('Time (s)')
    plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Run Times Graph.png'), dpi=600)
    plt.close()

    # Plot scatter graph of MSDs, with observed data on top
    line_colours = [ ]
    ax = plt.subplot(111)
    ax.plot(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', linewidth=1)
    ax.scatter(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', s=3)
    for ps_axis_value, ps_result_msds in zip(axis_values, result_msds):
        line = ax.plot(TIMEPOINTS, ps_result_msds, linewidth=0.5, label=str(ps_axis_value))
        line_colours.append(line[0].get_color())
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    leg = ax.legend(title=axis_label, loc='center left', bbox_to_anchor=(1.0, 0.5))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (µm^2)')
    plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'MSDs Graph.png'), bbox_inches='tight', dpi=667)
    plt.close()

    if do_individual_graphs:
        if not os.path.isdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, INDIVIDUAL_MSDS_DIR_NAME)):
            os.mkdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, INDIVIDUAL_MSDS_DIR_NAME))
        for ps_axis_value, ps_result_msds, ps_result_stds, ps_line_colour in zip(axis_values, result_msds, result_stds, line_colours):
            ax = plt.subplot(111)
            ax.plot(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', linewidth=1)
            try:
                ax.scatter(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', s=3)
                ax.plot(TIMEPOINTS, ps_result_msds, linewidth=0.5, label=str(ps_axis_value), color=ps_line_colour)
                error_lower = [ ps_result_msds[i] - ps_result_stds[i] for i in range(len(ps_result_msds)) ]
                error_upper = [ ps_result_msds[i] + ps_result_stds[i] for i in range(len(ps_result_msds)) ]
                ax.fill_between(TIMEPOINTS, error_lower, error_upper, alpha=0.25, facecolor=ps_line_colour)
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0)
                plt.xlabel('Time (s)')
                plt.ylabel('MSD (µm^2)')
                plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, INDIVIDUAL_MSDS_DIR_NAME, str(ps_axis_value) + '.png'), dpi=600)
            except Exception as e:
                pass
            plt.close()


def move_directory():
    if SHOULD_MOVE:
        shutil.move(os.path.join(RIP_DIR, RUN_NAME),
                    os.path.join(RS_DIR, RUN_NAME))


if __name__ == '__main__':
    setup()
    perform_checks()
    load_and_calculate_results()
    write_results()
    move_directory()
