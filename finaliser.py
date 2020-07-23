"""
Notes:
* Before running this script on Viking, be sure to load an appropriate Python 3 module,
  e.g. with the command `module load lang/SciPy-bundle/2020.03-foss-2019b-Python-3.7.4`
"""

import os
import sys
import math
import time
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
RIP_DIR = './runs_in_progress/'
RESULTS_DIR_NAME = 'RESULTS'
INDIVIDUAL_MSDS_DIR_NAME = 'Individual MSDs'
### Varaibles
CURRENT_DIR = ''


def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
    print()


def least_squares_score(timepoints, msds):
    point_deltas = [ ]
    for i in range(len(timepoints)):
        x = timepoints[i]
        true_msd = msds[i]
        ideal_msd = -0.1325*x**6 + 0.5272*x**5 - 0.8634*x**4 + 0.7492*x**3 - 0.3695*x**2 + 0.1029*x - 9E-05 # Function to approximate experimental data
        point_deltas.append(true_msd - ideal_msd)

    score = math.sqrt(sum([ x**2 for x in point_deltas ]))
    #score = 100 * max(0, (0.4 - score) / 0.4)

    return score


def setup():
    global CURRENT_DIR

    clear_screen()
    runs = sorted(next(os.walk(RIP_DIR))[1])
    print('Available runs:')
    for i in range(len(runs)):
        print(str(i+1) + ')', runs[i])
    print()
    chosen_dir = ''
    while not os.path.isdir(chosen_dir):
        try:
            choice = int(input('> '))
            chosen_dir = os.path.join(RIP_DIR, runs[choice-1])
        except:
            chosen_dir = ''
    print()
    CURRENT_DIR = chosen_dir

    if not os.path.isdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME)):
        os.mkdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME))


def perform_checks():
    # TODO:
    # 1) Check if any jobs are running
    # 2) Check if the finished jobs ran to completion
    # 3) Ask if should move to runs_saved directory
    return


def compile_results():
    output_files = sorted(os.listdir(os.path.join(CURRENT_DIR, 'output_files/')))
    xml_files = [ x for x in output_files if x[-4:] == '.xml' ]

    paramset_ids = set()
    for filename in xml_files:
        if 'paramset-' in filename:
            ps_id = int(filename.split('paramset-')[1].split('_')[0])
            paramset_ids.add(ps_id)
    paramset_ids = sorted(paramset_ids)

    ps_mean_simtime_means = [ ]
    ps_rms_simtime_stds = [ ]
    ps_mean_msds = [ ]
    ps_rms_stds = [ ]
    ps_scores = [ ]
    timepoints = [ ]

    for ps_num in paramset_ids:
        simtime_means = [ ]
        simtime_stds = [ ]
        msds_set = [ ]
        stds_set = [ ]

        ps_xml_files = [ x for x in xml_files if 'paramset-' + str(ps_num) in x ]
        for fn in ps_xml_files:
            # Parse original output XML file
            e = xml.etree.ElementTree.parse(os.path.join(CURRENT_DIR, 'output_files', fn)).getroot()

            # Load time statistics to array
            stats = e.find('STATISTICS').findall('STAT')
            for stat in stats:
                stat_name = [ value for key, value in stat.items() if key == 'NAME' ][0]
                stat_value = [ value for key, value in stat.items() if key == 'VALUE' ][0]
                if stat_name == 'MEAN_SIMULATION_DURATION':
                    simtime_means.append(float(stat_value))
                elif stat_name == 'STD_SIMULATION_DURATION':
                    simtime_stds.append(float(stat_value))

            # Load results to array
            results = e.find('RESULTS').findall('RES')
            msds = [ ]
            stds = [ ]
            for res in results:
                for key, value in res.items():
                    if key == 'MSD':
                        msds.append(float(value))
                    elif key == 'STD':
                        try:
                            stds.append(float(value))
                        except:
                            stds.append(0)
                    elif key == 'T':
                        if float(value) not in timepoints:
                            timepoints.append(float(value))
            msds_set.append(msds)
            stds_set.append(stds)

        # Calculate mean of simulation time means, and RMS of simulation time S.D.s, and save to the arrays
        mean_simtime_mean = 0
        rms_simtime_std = 0
        if len(ps_xml_files) > 0:
            mean_simtime_mean = sum(simtime_means) / len(simtime_means)
            rms_simtime_std = math.sqrt( sum([ x**2 for x in simtime_stds ]) / len(simtime_stds) )
        ps_mean_simtime_means.append(mean_simtime_mean)
        ps_rms_simtime_stds.append(rms_simtime_std)

        # Calculate mean of MSDs, and RMS of MSD S.D.s, and save to the arrays
        mean_msds = [ ]
        rms_stds = [ ]
        if len(ps_xml_files) > 0:
            num_timepoints = len(msds_set[0])
            for tp_index in range(num_timepoints):
                timepoint_msds = [ ]
                timepoint_stds = [ ]
                for set_num in range(len(msds_set)):
                    timepoint_msds.append(msds_set[set_num][tp_index])
                    timepoint_stds.append(stds_set[set_num][tp_index])
                mean_timepoint_msd = sum(timepoint_msds) / len(timepoint_msds)
                rms_timpoint_std = math.sqrt( sum([ x**2 for x in timepoint_stds ]) / len(timepoint_stds) )
                mean_msds.append(mean_timepoint_msd)
                rms_stds.append(rms_timpoint_std)
        ps_mean_msds.append(mean_msds)
        ps_rms_stds.append(rms_stds)

        # Calculate least squares score
        lss = 0
        if len(ps_xml_files) > 0:
            lss = least_squares_score(timepoints, mean_msds)
        ps_scores.append(lss)

    # Write run times to CSV
    with open(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Run Times.csv'), 'w') as outfile:
        outfile.write('Parameter Set,Run Time Mean (s),Run Time S.D. (s)\n')
        for i in range(len(paramset_ids)):
            outfile.write(str(paramset_ids[i]) + ',' + str(round(ps_mean_simtime_means[i], 9)) + ',' + str(round(ps_rms_simtime_stds[i], 9)) + '\n')

    # Write MSDs to CSV
    with open(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'MSDs.csv'), 'w') as outfile:
        outfile.write('Time (s),' + ','.join([ 'Set ' + str(x) + ',Set ' + str(x) for x in paramset_ids ]) + '\n')
        outfile.write(',' + ','.join(['MSD (µm^2),MSD S.D. (µm^2)'] * len(paramset_ids)) + '\n')
        for i in range(num_timepoints):
            try:
                outfile.write(str(timepoints[i]) + ',' + ','.join([ str(round(ps_mean_msds[x][i], 9)) + ',' + str(round(ps_rms_stds[x][i], 9)) for x in range(len(paramset_ids)) ]) + '\n')
            except Exception as e:
                break

    # Write least squares scores to CSV
    with open(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Scores.csv'), 'w') as outfile:
        outfile.write('Parameter Set,Least Squares Score)\n')
        for i in range(len(paramset_ids)):
            try:
                outfile.write(str(paramset_ids[i]) + ',' + str(round(ps_scores[i], 2)) + '\n')
            except Exception as e:
                break

    # Plot bar chart of least squares scores
    ax = plt.subplot(111)
    ax.bar(paramset_ids, ps_scores, width=0.5, align='center')
    #ax.set_ylim(0, 100)
    plt.xticks(paramset_ids, [ str(x) for x in paramset_ids ])
    plt.xlabel('Parameter Set')
    plt.ylabel('Least Squares Score')
    plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Scores Graph.png'), dpi=600)
    plt.close()

    # Get graph specifics
    ps_axis_label = ''
    ps_axis_values = [ ]
    ps_axis_value_type = ''
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
                discrepancies.append( (paramname, paramvals) )

    # Prompt for graph specifics
    clear_screen()
    if len(discrepancies) > 0:
        print('Discrepancies:')
        for i in range(len(discrepancies)):
            mm = discrepancies[i]
            if mm[1].count(mm[1][0]) == len(mm[1]):
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], 'All ' + str(mm[1][0]))
            else:
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], ', '.join(mm[1]))
    else:
        print('No discrepancies.')
    print()
    print('Enter parameter set title')
    ps_axis_label = input('> ')
    print()
    while True:
        print('Enter parameter set values (comma seperated, length ' + str(len(paramset_ids)) + ')')
        axis_labels_text = input('> ')
        ps_axis_values = [ x.strip() for x in axis_labels_text.split(',') ]
        if len(ps_axis_values) == len(paramset_ids):
            break
        print('Wrong length.')
    try:
        [ float(x) for x in ps_axis_values ]
        print()
        print('Use proportional axis for runtimes?')
        if input('> ').upper() in [ 'Y', 'YES' ]:
            ps_axis_values = [ float(x) for x in ps_axis_values ]
            ps_axis_value_type = 'Number'
        else:
            ps_axis_value_type = 'String'
    except:
        print('Axis labels will be strings.')
        ps_axis_value_type = 'String'
    print()
    if ps_axis_value_type == 'Number':
        print('Perform linear regression on runtimes?')
        do_linreg = input('> ').upper() in [ 'Y', 'YES' ]
    print()
    print('Produce individual graphs for each parameter set?')
    do_individual_graphs = input('> ').upper() in [ 'Y', 'YES' ]
    print()
    print('Finalising results...')
    print()

    # Linear regression fit for run times
    if do_linreg:
        slope, intercept, r_value, p_value, std_err = scistats.linregress(ps_axis_values, ps_mean_simtime_means)

    # Plot scatter graph of run times with error bars, with linear regression fit on top
    ax = plt.subplot(111)
    ax.errorbar(ps_axis_values, ps_mean_simtime_means, yerr=ps_rms_simtime_stds, fmt='o', linestyle='None')
    if do_linreg:
        xs = np.array(ps_axis_values)
        ys = slope * xs + intercept
        ax.plot(xs, ys, '-r', color='black', label='y = ' + str(round(slope, 1)) + 'x + ' + str(round(intercept, 1)) + ', R2 = ' + str(round(r_value**2, 3)))
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
        leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
    plt.xlabel(ps_axis_label)
    plt.ylabel('Time (s)')
    plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'Run Times Graph.png'), dpi=600)
    plt.close()

    # Plot scatter graph of MSDs, with observed data on top
    line_colours = [ ]
    ax = plt.subplot(111)
    ax.plot(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', linewidth=1)
    ax.scatter(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', s=3)
    for i in range(len(ps_axis_values)):
        try:
            line = ax.plot(timepoints, ps_mean_msds[i], linewidth=0.5, label=str(ps_axis_values[i]))
            line_colours.append(line[0].get_color())
        except Exception as e:
            pass
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    leg = ax.legend(title=ps_axis_label, loc='center left', bbox_to_anchor=(1.0, 0.5))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (µm^2)')
    plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, 'MSDs Graph.png'), dpi=600)
    plt.close()

    if do_individual_graphs:
        if not os.path.isdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, INDIVIDUAL_MSDS_DIR_NAME)):
            os.mkdir(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, INDIVIDUAL_MSDS_DIR_NAME))
        for i in range(len(ps_axis_values)):
            ax = plt.subplot(111)
            ax.plot(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', linewidth=1)
            try:
                ax.scatter(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', s=3)
                ax.plot(timepoints, ps_mean_msds[i], linewidth=0.5, label=str(ps_axis_values[i]), color=line_colours[i])
                error_lower = [ ps_mean_msds[i][j] - ps_rms_stds[i][j] for j in range(len(ps_mean_msds[i])) ]
                error_upper = [ ps_mean_msds[i][j] + ps_rms_stds[i][j] for j in range(len(ps_mean_msds[i])) ]
                ax.fill_between(timepoints, error_lower, error_upper, alpha=0.25, facecolor=line_colours[i])
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0)
                plt.xlabel('Time (s)')
                plt.ylabel('MSD (µm^2)')
                plt.savefig(os.path.join(CURRENT_DIR, RESULTS_DIR_NAME, INDIVIDUAL_MSDS_DIR_NAME, str(ps_axis_values[i]) + '.png'), dpi=600)
            except Exception as e:
                pass
            plt.close()


if __name__ == '__main__':
    setup()
    perform_checks()
    compile_results()
