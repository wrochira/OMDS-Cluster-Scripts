# -*- coding: utf-8 -*-

"""
Notes:
* Before running this script on Viking, be sure to load an appropriate Python 3 module
* What SLURM refers to as the 'array job ID' (ArrayJobID) is referred to in this script as 'job group ID'
"""

import sys
if sys.version_info[0] < 3:
    print('You must load a compatible Python module before running this script.')
    print('One such module can be loaded with the following command:')
    print('\tmodule load data/scikit-learn/0.20.2-foss-2018b-Python-3.6.6\n')
    exit(1)

import os
import csv
import math
import shutil
import datetime
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as scistats
matplotlib.rcParams.update({'errorbar.capsize': 5})


BINARIES_DIR = './binaries/'
PARAMSETS_DIR = './parameter_sets/'
SIMULATIONS_DIR = './simulations/'
OUTPUT_DIR_NAME = 'output_files'
RESULTS_DIR_NAME = 'RESULTS'
INDIVIDUAL_MSDS_DIR_NAME = 'Individual MSDs'
PARAMSETS_FILE_NAME = 'parameter_sets.xml'
JOB_SET_INFO_FILE_NAME = 'job_set_info.xml'
DISCREPANCIES_FILE_NAME = 'variables.txt'
CLF_RUNNING = 'RUNNING'
CLF_PENDING = 'PENDING'
CLF_FINISHED = 'FINISHED'
OBS_TIMEPOINTS = [ 0, 0.03333, 0.06666, 0.09999, 0.13332, 0.16665, 0.19998, 0.23331, 0.26664, 0.29997, 0.3333, 0.36663, 0.39996, 0.43329, 0.46662, 0.49995, 0.53328, 0.56661, 0.59994, 0.63327, 0.6666, 0.69993, 0.73326, 0.76659, 0.79992, 0.83325, 0.86658, 0.89991, 0.93324, 0.96657, 0.9999 ]
OBS_DATAPOINTS = [ 0, 0.0029767, 0.0050122, 0.0072264, 0.0086977, 0.009889, 0.010522, 0.010981, 0.011506, 0.012154, 0.012248, 0.012361, 0.012455, 0.012771, 0.012979, 0.013139, 0.013295, 0.013527, 0.013463, 0.013404, 0.013382, 0.013477, 0.013626, 0.013696, 0.013713, 0.01374, 0.013832, 0.013819, 0.013804, 0.013799, 0.013784 ]

JOB_SETS = { }


def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
    print()


def title(page_name):
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    padding_left = math.floor((terminal_width - len(page_name) - 2) / 2)
    padding_right = math.ceil((terminal_width - len(page_name) - 2) / 2)
    print('='*padding_left, page_name, '='*padding_right)
    date_string = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    padding_left = math.floor((terminal_width - len(date_string) - 2) / 2)
    padding_right = math.ceil((terminal_width - len(date_string) - 2) / 2)
    print('='*padding_left, date_string, '='*padding_right)
    print()


def print_table(rows, col_widths):
    num_cols = max(len(row) for row in rows)
    col_widths = list(col_widths)
    while len(col_widths) < num_cols:
        col_widths.append(col_widths[-1])
    full_width = col_widths[0] + col_widths[1] * (num_cols-1)
    for row in rows:
        line = ''
        for col_width, value in zip(col_widths, row):
            line += (str(value)[::-1] + ' '*col_width)[:col_width][::-1]
        print(line)
    print()


def least_squares_score(timepoints, msds):
    point_deltas = [ ]
    for x, true_msd in zip(timepoints, msds):
        if 0 < x < 1:
            ideal_msd = -0.1325*x**6 + 0.5272*x**5 - 0.8634*x**4 + 0.7492*x**3 - 0.3695*x**2 + 0.1029*x - 9E-05 # Function to approximate experimental data
            point_deltas.append(true_msd - ideal_msd)
    score = math.sqrt(sum([ x**2 for x in point_deltas ]))
    return score


def setup_environment():
    for dir_name in (BINARIES_DIR, PARAMSETS_DIR, SIMULATIONS_DIR):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)


def update_job_set_data():
    global JOB_SETS
    print()
    print('Loading...')
    job_ids = { } # Keyed by job allocation number
    # For each job-set directory...
    for job_set_name in os.listdir(SIMULATIONS_DIR):
        job_set_path = os.path.join(SIMULATIONS_DIR, job_set_name)
        if not os.path.isdir(job_set_path):
            continue
        JOB_SETS[job_set_name] = { }
        # Check if analysis has already been performed, and skip it if so
        if RESULTS_DIR_NAME in os.listdir(job_set_path):
            JOB_SETS[job_set_name]['classification'] = CLF_FINISHED
        # Get job IDs and run distribution info
        job_group_ids = [ ]
        paramset_title = None
        instances_per_paramset = None
        runs_per_instance = None
        instance_time_limit = None
        try:
            tree = ET.parse(os.path.join(job_set_path, JOB_SET_INFO_FILE_NAME))
            root = tree.getroot()
            for element in root:
                if element.tag == 'JobGroups':
                    for job_group in element:
                        job_group_id = int(job_group.get('id'))
                        job_group_ids.append(job_group_id)
                elif element.tag == 'Parameter':
                    name, value = element.get('name'), element.get('value')
                    if name == 'ParamsetTitle':
                        paramset_title = value
                    if name == 'InstancesPerParamset':
                        instances_per_paramset = int(value)
                    if name == 'RunsPerInstance':
                        runs_per_instance = int(value)
                    if name == 'InstanceTimeLimit':
                        instance_time_limit = int(value)
        except:
            print('Error parsing', JOB_SET_INFO_FILE_NAME, 'for job-set', job_set_name)
            exit(1)
        jobs = { } # Keyed by job ID
        for job_group_id in job_group_ids:
            for instance_id in range(instances_per_paramset):
                job_id = str(job_group_id) + '_' + str(instance_id)
                paramset_id = job_group_ids.index(job_group_id)
                jobs[job_id] = { 'job_group_id' : job_group_id,
                                 'paramset_id' : paramset_id,
                                 'instance_id' : instance_id,
                                 'job_alloc_num' : None,
                                 'state' : None,
                                 'time_elapsed' : 0,
                                 'runs_completed' : 0 }
        # Get SLURM info for each job (instance)
        job_group_ids = sorted(set([ x['job_group_id'] for x in jobs.values() ]))
        job_group_ids_str = ','.join(str(x) for x in job_group_ids)
        p = subprocess.Popen([ 'sacct', '-j', job_group_ids_str, '-o', 'JobID,JobIDRaw,State,ElapsedRaw,TimelimitRaw', '-P', '-X', '--noheader' ], stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        for line in stdout.decode().split('\n'):
            if len(line) == 0:
                continue
            job_id, job_alloc_num_str, state, time_elapsed_str, time_limit_str = line.strip().split('|')
            state = state.split(' ')[0] # First word is enough
            # Deal with unallocated ID ranges
            if '[' in job_id:
                job_group_id_str, job_instance_id_str = job_id.split('_')
                instance_id_range = [ int(x) for x in job_instance_id_str[1:-1].split('-') ]
                for instance_id in range(instance_id_range[0], instance_id_range[1]+1):
                    job_id = job_group_id_str + '_' + str(instance_id)
                    jobs[job_id]['state'] = state
                    jobs[job_id]['time_elapsed'] = int(time_elapsed_str)
            else:
                job_alloc_num = int(job_alloc_num_str)
                jobs[job_id]['job_alloc_num'] = job_alloc_num
                jobs[job_id]['state'] = state
                jobs[job_id]['time_elapsed'] = int(time_elapsed_str)
                job_ids[job_alloc_num] = job_id
        # Count completed runs for each instance
        for log_name in os.listdir(os.path.join(job_set_path, 'output_std')):
            if not log_name.lower().endswith('.log'):
                continue
            job_alloc_num = int(log_name[4:-4])
            job_id = job_ids[job_alloc_num]
            log_path = os.path.join(job_set_path, 'output_std', log_name)
            with open(log_path, 'r') as infile:
                for line in infile.readlines():
                    if line.strip().endswith('run completed'):
                        num = int(line.strip().split(' ')[2])
                        if jobs[job_id]['runs_completed'] is None or \
                           num > jobs[job_id]['runs_completed']:
                            jobs[job_id]['runs_completed'] = num
        # Determine job-set classification
        classification = None
        job_states = set([ x['state'] for x in jobs.values() ])
        if CLF_PENDING in job_states or CLF_RUNNING in job_states:
            classification = CLF_RUNNING
        else:
            classification = CLF_PENDING
        JOB_SETS[job_set_name]['jobs'] = jobs
        JOB_SETS[job_set_name]['classification'] = classification
        JOB_SETS[job_set_name]['paramset_title'] = paramset_title
        JOB_SETS[job_set_name]['num_paramsets'] = len(job_group_ids)
        JOB_SETS[job_set_name]['instances_per_paramset'] = instances_per_paramset
        JOB_SETS[job_set_name]['runs_per_instance'] = runs_per_instance
        JOB_SETS[job_set_name]['instance_time_limit'] = instance_time_limit


def main_menu():
    update_job_set_data()
    num_running, num_pending, num_finished = 0, 0, 0
    for job_set in JOB_SETS.values():
        if job_set['classification'] == CLF_RUNNING:
            num_running += 1
        elif job_set['classification'] == CLF_PENDING:
            num_pending += 1
        elif job_set['classification'] == CLF_FINISHED:
            num_finished += 1
    clear_screen()
    title('MAIN MENU')
    print(num_running, 'job-sets in progress')
    print(num_pending, 'job-sets pending analysis')
    print(num_finished, 'job-sets finished')
    print()
    print('Choose an option:')
    print('1) Queue new job-set')
    print('2) Monitor running job-sets')
    print('3) Analyse finished job-sets')
    print('4) Cancel running job-sets')
    print('Q) Quit')
    choice = None
    options = set(str(x+1) for x in range(4))
    while choice not in options:
        choice = input('> ').strip().upper()
        if choice == 'Q':
            clear_screen()
            exit(0)
    choice_index = int(choice)-1
    chosen_function = [ queue, monitor, analyse, cancel ][choice_index]
    chosen_function()


def queue():
    clear_screen()
    title('QUEUE')
    print('Choose a parameter sets file:')
    paramsets_file_names = sorted([ name for name in os.listdir(PARAMSETS_DIR) if name.lower().endswith('.xml') ])
    for i, paramsets_file_name in enumerate(paramsets_file_names):
        print(str(i+1) + ')', paramsets_file_name)
    print('M) Back to main menu')
    choice = None
    options = set(str(x+1) for x in range(len(paramsets_file_names)))
    while choice not in options:
        choice = input('> ').strip().upper()
        if choice == 'M':
            return
    choice_index = int(choice)-1
    # Get configuration from user
    print()
    print('How many instances per parameter set?')
    instances_per_paramset = int(input('> ').strip())
    print()
    print('How many runs per instance?')
    runs_per_instance = int(input('> ').strip())
    total_runs = instances_per_paramset * runs_per_instance
    print()
    print('Total number of runs will be', str(total_runs) + ',', 'split over', instances_per_paramset, 'instance(s) per parameter set.')
    print()
    print('Run time limit (hh:mm:ss)')
    run_time_str = input('> ').strip()
    instance_time_limit = sum(int(x) * 60**i for i, x in enumerate(reversed(run_time_str.split(':'))))
    print()
    print('Partition name (leave blank for default: \'nodes\')')
    partition_name = input('> ').strip()
    print()
    print('Email alerts?')
    do_alerts = input('> ').strip().upper() in ['Y', 'YES']
    print()
    if do_alerts:
        print()
        print('Email address')
        email = input('> ').strip()
    # Generate run directory
    run_name = datetime.datetime.now().strftime('%y%m%d%H%M')
    run_dir = os.path.join(SIMULATIONS_DIR, run_name)
    os.mkdir(run_dir)
    os.mkdir(os.path.join(run_dir, 'output_files/'))
    os.mkdir(os.path.join(run_dir, 'output_std/'))
    # Copy latest binary
    binaries = sorted([ x for x in os.listdir(BINARIES_DIR) if x[-4:] == '.jar' ])
    latest_binary = binaries[-1]
    shutil.copy2(os.path.join(BINARIES_DIR, latest_binary), run_dir)
    # Parse default parameter sets file
    paramset_values_default = { }
    paramsets_file_name = sorted([ x for x in paramsets_file_names if 'default' in x and x.lower().endswith('.xml') ])[-1]
    paramsets_file_path =  os.path.join(PARAMSETS_DIR, paramsets_file_name)
    paramsets_file_content = open(paramsets_file_path, 'r').read()
    start_index = paramsets_file_content.index('<EXPERIMENT>')
    end_index = paramsets_file_content.index('</EXPERIMENT>')
    paramsets_file_content = paramsets_file_content[start_index:end_index+13]
    root = ET.fromstring(paramsets_file_content)
    num_paramsets = len(root.findall('PARAMETERS'))
    for paramset in root:
        for param in paramset:
            name, value = param.get('NAME'), param.get('VALUE')
            paramset_values_default[name] = [ value ]
    # Parse, edit, and export selected parameter sets file
    paramset_values_chosen = { }
    paramsets_file_name = paramsets_file_names[choice_index]
    paramsets_file_path = os.path.join(PARAMSETS_DIR, paramsets_file_name)
    paramsets_file_content = open(paramsets_file_path, 'r').read()
    start_index = paramsets_file_content.index('<EXPERIMENT>')
    end_index = paramsets_file_content.index('</EXPERIMENT>')
    paramsets_file_content = paramsets_file_content[start_index:end_index+13]
    root = ET.fromstring(paramsets_file_content)
    num_paramsets = len(root.findall('PARAMETERS'))
    for paramset in root:
        for param in paramset:
            name, value = param.get('NAME'), param.get('VALUE')
            if name == 'N_PROTEINS':
                param.set('VALUE', str(runs_per_instance))
            if name not in paramset_values_chosen:
                paramset_values_chosen[name] = [ ]
            paramset_values_chosen[name].append(value)
    paramsets_xml = ET.tostring(root, encoding='utf8', method='xml')
    with open(os.path.join(run_dir, PARAMSETS_FILE_NAME), 'wb') as outfile:
        outfile.write(paramsets_xml)
    # Find discrepancies and write them to file
    with open(os.path.join(run_dir, DISCREPANCIES_FILE_NAME), 'w') as outfile:
        for key, value_d in paramset_values_default.items():
            if key == 'N_PROTEINS':
                continue
            if key not in paramset_values_chosen.keys():
                print('ERROR: Missing parameter sets file is missing required parameter:', key)
                exit(1)
            value_c = paramset_values_chosen[key]
            if set(value_d) != set(value_c):
                outfile.write(key + ': ' + ', '.join(value_c) + '\n')
    # Generate jobscript
    with open(os.path.join(run_dir, 'jobscript.sh'), 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('#SBATCH --job-name=omd_sim\n')
        if do_alerts:
            outfile.write('#SBATCH --mail-type=ALL\n')
            outfile.write('#SBATCH --mail-user=' + email + '\n')
        if partition_name != '':
            outfile.write('#SBATCH --partition=' + partition_name + '\n')
        outfile.write('#SBATCH --cpus-per-task=8\n')
        outfile.write('#SBATCH --mem=4gb\n')
        outfile.write('#SBATCH --time=' + run_time_str + '\n')
        outfile.write('#SBATCH --output=./output_std/omds%j.log\n')
        outfile.write('#SBATCH --error=./output_std/omds%j.err\n')
        outfile.write('#SBATCH --account=biol-stdbom-2019\n')
        outfile.write('#SBATCH --array=0-' + str(instances_per_paramset-1) + '\n')
        outfile.write('module load lang/Java/1.8.0_212\n')
        outfile.write('export MALLOC_ARENA_MAX=8\n')
        outfile.write('vmArgs="-Xmx1G -XX:ParallelGCThreads=1 -jar"\n')
        outfile.write('java $vmArgs ./' + latest_binary + ' ' + PARAMSETS_FILE_NAME + ' ./output_files $PSET_ID $SLURM_ARRAY_TASK_ID')
    # Generate launcher script
    with open(os.path.join(run_dir, 'launcher.sh'), 'w') as outfile:
        outfile.write('cd "${0%/*}"\n') # Sets working directory to script directory
        for paramset_id in range(num_paramsets):
            outfile.write('sbatch --export=PSET_ID=' + str(paramset_id) + ' jobscript.sh\n')
    # Launch the tasks
    print('Ready to launch.')
    input('> ')
    print()
    p = subprocess.Popen(['sh', run_dir + '/launcher.sh'], stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    # Parse output and save job-set info file
    job_group_ids = [ int(x.strip()) for x in stdout.decode().split('Submitted batch job ') if len(x) > 0 ]
    if len(job_group_ids) > 0:
        root = ET.Element('JobSet')
        job_groups = ET.SubElement(root, 'JobGroups')
        for job_group_id in job_group_ids:
            ET.SubElement(job_groups, 'JobGroup', id=str(job_group_id))
        ET.SubElement(root, 'Parameter', name='ParamsetTitle', value=paramsets_file_name[:-4])
        ET.SubElement(root, 'Parameter', name='InstancesPerParamset', value=str(instances_per_paramset))
        ET.SubElement(root, 'Parameter', name='RunsPerInstance', value=str(runs_per_instance))
        ET.SubElement(root, 'Parameter', name='InstanceTimeLimit', value=str(instance_time_limit))
        tree = ET.ElementTree(root)
        tree.write(os.path.join(run_dir, JOB_SET_INFO_FILE_NAME))
        input('Done. Press any key to continue.')
    else:
        print('Launch failed.\n')
        exit(1)


def monitor():
    clear_screen()
    title('MONITOR')
    print('Choose a job-set:')
    running_job_set_names = [ ]
    for job_set_name, job_set in JOB_SETS.items():
        if job_set['classification'] == CLF_RUNNING:
            running_job_set_names.append(job_set_name)
    running_job_set_names.sort()
    for i, job_set_name in enumerate(running_job_set_names):
        job_set_title = str(JOB_SETS[job_set_name]['paramset_title'])
        print(str(i+1) + ')', job_set_name, '(' + job_set_title + ')')
    print('M) Back to main menu')
    choice = None
    options = set(str(x+1) for x in range(len(running_job_set_names)))
    while choice not in options:
        choice = input('> ').strip().upper()
        if choice == 'M':
            return
    choice_index = int(choice)-1
    job_set_name = running_job_set_names[choice_index]
    job_set = JOB_SETS[job_set_name]
    jobs = job_set['jobs']
    while True:
        clear_screen()
        title('MONITOR')
        num_paramsets = max(x['paramset_id'] for x in jobs.values())+1
        num_instances = max(x['instance_id'] for x in jobs.values())+1
        paramset_ids = list(range(num_paramsets))
        instance_ids = list(range(num_instances))
        state_matrix = [ [ None for _ in paramset_ids ] for _ in instance_ids ]
        runs_matrix = [ [ None for _ in paramset_ids ] for _ in instance_ids ]
        paramset_runs_completed = [ 0 for _ in paramset_ids ]
        paramset_runs_target = [ 0 for _ in paramset_ids ]
        paramset_hours_elapsed = [ 0 for _ in paramset_ids ]
        paramset_hours_limit = [ 0 for _ in paramset_ids ]
        for job in jobs.values():
            state_matrix[job['instance_id']][job['paramset_id']] = job['state'][0]
            runs_matrix[job['instance_id']][job['paramset_id']] = job['runs_completed']
            paramset_runs_completed[job['paramset_id']] += job['runs_completed']
            paramset_runs_target[job['paramset_id']] += job_set['runs_per_instance']
            paramset_hours_elapsed[job['paramset_id']] += job['time_elapsed'] / 3600
            paramset_hours_limit[job['paramset_id']] += job_set['instance_time_limit'] / 3600
        paramset_hours_remaining = [ ]
        paramset_hours_projected = [ ]
        paramset_on_track = [ ]
        for runs_completed, runs_target, hours_elapsed, hours_limit in zip(paramset_runs_completed, paramset_runs_target, paramset_hours_elapsed, paramset_hours_limit):
            hours_remaining = hours_limit - hours_elapsed
            paramset_hours_remaining.append(hours_remaining)
            if hours_elapsed == 0 or runs_completed == 0:
                hours_projected = 'N/A'
                on_track = 'N/A'
            else:
                hours_projected = hours_elapsed * (runs_target/runs_completed-1) 
                on_track = hours_projected <= hours_remaining
            paramset_hours_projected.append(hours_projected)
            paramset_on_track.append(on_track)
        rows = [ ]
        rows.append([ '' ] + [ 'PS' + str(paramset_id) for paramset_id in paramset_ids ])
        rows.append([ 'Instance ID' ])
        for instance_id, states, runs in zip(instance_ids, state_matrix, runs_matrix):
            rows.append([ instance_id ] + [ s + ':' + str(r) for s, r in zip(states, runs) ])
        rows.append([ ])
        rows.append([ 'Runs Total' ] + paramset_runs_target)
        rows.append([ 'Hours Total' ] + [ round(x, 1) for x in paramset_hours_limit ])
        rows.append([ ])
        rows.append([ 'Runs Complete' ] + paramset_runs_completed)
        rows.append([ 'Hours Elapsed' ] + [ round(x, 1) for x in paramset_hours_elapsed ])
        rows.append([ 'Hours Projected' ] + [ round(x, 1) if type(x) == float else x for x in paramset_hours_projected ])
        rows.append([ 'Hours Remaining' ] + [ round(x, 1) for x in paramset_hours_remaining ])
        rows.append([ ])
        rows.append([ 'On Track?' ] + paramset_on_track)
        print_table(rows, (15,8))
        print('R) Refresh page')
        print('M) Back to main menu')
        while True:
            choice = input('> ').strip().upper()
            if choice == 'R':
                update_job_set_data()
                break
            if choice == 'M':
                return


def analyse():
    clear_screen()
    title('ANALYSE')
    print('Choose a job-set:')
    finished_job_set_names = [ ]
    for job_set_name, job_set in JOB_SETS.items():
        if job_set['classification'] in (CLF_PENDING, CLF_FINISHED):
            finished_job_set_names.append(job_set_name)
    finished_job_set_names.sort()
    for i, job_set_name in enumerate(finished_job_set_names):
        job_set_title = str(JOB_SETS[job_set_name]['paramset_title'])
        print(str(i+1) + ')', job_set_name, '(' + job_set_title + ')')
    print('M) Back to main menu')
    choice = None
    options = set(str(x+1) for x in range(len(finished_job_set_names)))
    while choice not in options:
        choice = input('> ').strip().upper()
        if choice == 'M':
            return
    choice_index = int(choice)-1
    job_set_name = finished_job_set_names[choice_index]
    job_set = JOB_SETS[job_set_name]
    jobs = job_set['jobs']
    num_paramsets = job_set['num_paramsets']
    run_dir = os.path.join(SIMULATIONS_DIR, job_set_name)
    # Get relevant output files
    all_output_files = sorted(os.listdir(os.path.join(run_dir, OUTPUT_DIR_NAME)))
    xml_output_files = [ x for x in all_output_files if x[-4:] == '.xml' ]
    incomplete_file_names = set()
    results = { }
    for paramset_id in range(num_paramsets):
        paramset_results = { 'n' : 0,
                             'result_msds' : { },
                             'result_stds' : { },
                             'timing_mean' : None,
                             'timing_std' : None,
                             'lss' : None }
        paramset_file_results = [ ]
        # Get data from all available output files for each parameter set
        paramset_output_files = [ x for x in xml_output_files if ('paramset-' + str(paramset_id) + '_') in x ]
        for filename in paramset_output_files:
            filepath = os.path.join(run_dir, OUTPUT_DIR_NAME, filename)
            root = ET.parse(filepath).getroot()
            file_results = { 'n' : 0,
                             'result_msds' : { },
                             'result_stds' : { },
                             'timing_mean' : None,
                             'timing_std' : None }
            # Load results
            try:
                for res in root.find('RESULTS').findall('RES'):
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
                stats = root.find('STATISTICS').findall('STAT')
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
            paramset_file_results.append(file_results)
        # Count total number of runs
        for file_results in paramset_file_results:
            paramset_results['n'] += file_results['n']
        # Skip parameter sets with no successful runs
        if paramset_results['n'] == 0:
            continue
        # Calculate weighted average of timings
        timing_mean = 0
        timing_std = 0
        timepoints = set()
        for file_results in paramset_file_results:
            timing_mean += file_results['n'] * file_results['timing_mean']
            timing_std += file_results['n'] * file_results['timing_std']**2
            timepoints.update(set(file_results['result_msds'].keys()))
        timing_mean /= paramset_results['n']
        timing_std = math.sqrt(timing_std / paramset_results['n'])
        paramset_results['timing_mean'] = timing_mean
        paramset_results['timing_std'] = timing_std
        paramset_results['timepoints'] = sorted(list(timepoints))
        # Calculate weighted average of results
        for timepoint in paramset_results['timepoints']:
            results_msd = 0
            results_std = 0
            for file_results in paramset_file_results:
                results_msd += file_results['n'] * file_results['result_msds'][timepoint]
                results_std += file_results['n'] * file_results['result_stds'][timepoint]**2
            results_msd /= paramset_results['n']
            results_std = math.sqrt(results_std / paramset_results['n'])
            paramset_results['result_msds'][timepoint] = results_msd
            paramset_results['result_stds'][timepoint] = results_std
        # Calculate LSS
        paramset_results['lss'] = least_squares_score(list(paramset_results['result_msds'].keys()),
                                                      list(paramset_results['result_msds'].values()))
        results[paramset_id] = paramset_results
    # Report on the meta-analyses
    if len(incomplete_file_names) > 0:
        print('WARNING: Some instances did not complete enough runs to be included in the analyses:')
        for filename in sorted(incomplete_file_names):
            print('*', filename)
    null_paramset_ids = set(range(num_paramsets)) - set(results.keys())
    if len(null_paramset_ids) == num_paramsets:
        print()
        print('ERROR: None of the parameter sets completed any runs; there is nothing to analyse.')
        print()
        print('Press any key to return to the main menu.')
        input('> ')
        return
    elif len(null_paramset_ids) > 0:
        print()
        print('WARNING: Some parameter sets did not complete any runs:')
        for paramset_id in sorted(null_paramset_ids):
            print('* PS' + str(paramset_id))
    print()
    print('The run counts for each parameter set are:')
    for paramset_id in range(num_paramsets):
        if paramset_id in results.keys():
            print('* PS' + str(paramset_id) + ':', results[paramset_id]['n'])
        else:
            print('* PS' + str(paramset_id) + ': NONE')
    print()
    print('Press any key to continue.')
    input('> ')
    # Create the results directory
    results_dir = os.path.join(run_dir, RESULTS_DIR_NAME)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    paramset_ids = list(results.keys())
    timepoints = [ x['timepoints'] for x in results.values() ]
    result_msds = [ list(x['result_msds'].values()) for x in results.values() ]
    result_stds = [ list(x['result_stds'].values()) for x in results.values() ]
    runtime_means = [ x['timing_mean'] for x in results.values() ]
    runtime_stds = [ x['timing_std'] for x in results.values() ]
    least_squares_scores = [ x['lss'] for x in results.values() ]
    # Write run times to CSV
    with open(os.path.join(results_dir, 'Run Times.csv'), 'w') as outfile:
        outfile.write('Parameter Set,Run Time Mean (s),Run Time S.D. (s)\n')
        for paramset_id, paramset_results in results.items():
            outfile.write(str(paramset_id) + ',' + str(round(paramset_results['timing_mean'], 9)) + ',' + str(round(paramset_results['timing_std'], 9)) + '\n')
    # Write MSDs to CSV
    with open(os.path.join(results_dir, 'MSDs.csv'), 'w') as outfile:
        outfile.write('Time (s),' + ','.join([ 'Set ' + str(x) + ',Set ' + str(x) for x in paramset_ids ]) + '\n')
        outfile.write(',' + ','.join(['MSD (µm^2),MSD S.D. (µm^2)'] * len(results.keys())) + '\n')
        all_timepoints = sorted(list(set([ a for b in timepoints for a in b ])))
        for timepoint in all_timepoints:
            line_values = [ str(timepoint) ]
            for paramset_results in results.values():
                msd = str(round(paramset_results['result_msds'][timepoint], 9)) if timepoint in paramset_results['result_msds'].keys() else '-'
                std = str(round(paramset_results['result_stds'][timepoint], 9)) if timepoint in paramset_results['result_stds'].keys() else '-'
                line_values += [ msd, std ]
            outfile.write(','.join(line_values) + '\n')
    # Write least squares scores to CSV
    with open(os.path.join(results_dir, 'Scores.csv'), 'w') as outfile:
        outfile.write('Parameter Set,Least Squares Score)\n')
        for paramset_id, paramset_results in results.items():
            outfile.write(str(paramset_id) + ',' + str(round(paramset_results['lss'], 2)) + '\n')
    # Plot bar chart of least squares scores
    ax = plt.subplot(111)
    ax.bar(paramset_ids, least_squares_scores, width=0.5, align='center')
    #ax.set_ylim(0, 100)
    plt.xticks(paramset_ids, [ str(x) for x in paramset_ids ])
    plt.xlabel('Parameter Set')
    plt.ylabel('Least Squares Score')
    plt.savefig(os.path.join(results_dir, 'Scores Graph.png'), dpi=600)
    plt.close()
    # Get graph specifics
    axis_label = ''
    axis_values = [ ]
    axis_value_type = ''
    do_individual_graphs = False
    do_linreg = False
    # Load discrepancies from file
    discrepancies = [ ]
    try:
        with open(os.path.join(run_dir, DISCREPANCIES_FILE_NAME), 'r') as infile:
            for line in infile.readlines():
                line = line.strip()
                if line:
                    paramname = line.split(': ')[0]
                    paramvals = line.split(': ')[1].split(', ')
                    discrepancies.append((paramname, paramvals))
    except:
        print('ERROR: Failed to read', DISCREPANCIES_FILE_NAME)
        print()
        print('Press any key to return to the main menu.')
        input('> ')
        return
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
    print('Analysing results...')
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
    plt.savefig(os.path.join(results_dir, 'Run Times Graph.png'), dpi=600)
    plt.close()
    # Plot scatter graph of MSDs, with observed data on top
    line_colours = [ ]
    ax = plt.subplot(111)
    ax.plot(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', linewidth=1)
    ax.scatter(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', s=3)
    for ps_axis_value, ps_timepoints, ps_result_msds in zip(axis_values, timepoints, result_msds):
        line = ax.plot(ps_timepoints, ps_result_msds, linewidth=0.5, label=str(ps_axis_value))
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
    plt.savefig(os.path.join(results_dir, 'MSDs Graph.png'), bbox_inches='tight', dpi=667)
    plt.close()
    if do_individual_graphs:
        if not os.path.isdir(os.path.join(results_dir, INDIVIDUAL_MSDS_DIR_NAME)):
            os.mkdir(os.path.join(results_dir, INDIVIDUAL_MSDS_DIR_NAME))
        for ps_axis_value, ps_timepoints, ps_result_msds, ps_result_stds, ps_line_colour in zip(axis_values, timepoints, result_msds, result_stds, line_colours):
            ax = plt.subplot(111)
            ax.plot(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', linewidth=1)
            try:
                ax.scatter(OBS_TIMEPOINTS, OBS_DATAPOINTS, color='black', s=3)
                ax.plot(ps_timepoints, ps_result_msds, linewidth=0.5, label=str(ps_axis_value), color=ps_line_colour)
                error_lower = [ ps_result_msds[i] - ps_result_stds[i] for i in range(len(ps_result_msds)) ]
                error_upper = [ ps_result_msds[i] + ps_result_stds[i] for i in range(len(ps_result_msds)) ]
                ax.fill_between(ps_timepoints, error_lower, error_upper, alpha=0.25, facecolor=ps_line_colour)
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0)
                plt.xlabel('Time (s)')
                plt.ylabel('MSD (µm^2)')
                plt.savefig(os.path.join(results_dir, INDIVIDUAL_MSDS_DIR_NAME, str(ps_axis_value) + '.png'), dpi=600)
            except Exception as e:
                pass
            plt.close()
    input('Done. Press any key to continue.')


def cancel():
    clear_screen()
    title('CANCEL')
    print('Choose a job-set:')
    running_job_set_names = [ ]
    for job_set_name, job_set in JOB_SETS.items():
        if job_set['classification'] == CLF_RUNNING:
            running_job_set_names.append(job_set_name)
    running_job_set_names.sort()
    for i, job_set_name in enumerate(running_job_set_names):
        job_set_title = str(JOB_SETS[job_set_name]['paramset_title'])
        print(str(i+1) + ')', job_set_name, '(' + job_set_title + ')')
    print('M) Back to main menu')
    choice = None
    options = set(str(x+1) for x in range(len(running_job_set_names)))
    while choice not in options:
        choice = input('> ').strip().upper()
        if choice == 'M':
            return
    choice_index = int(choice)-1
    job_set_name = running_job_set_names[choice_index]
    job_set = JOB_SETS[job_set_name]
    jobs = job_set['jobs']
    job_group_ids = set([ job['job_group_id'] for job in jobs.values() ])
    print()
    print('The following job groups will be cancelled:')
    for job_group_id in job_group_ids:
        print('*', job_group_id)
    print()
    print('Are you sure you want to do this?')
    should_continue = input('> ').strip().upper() in ['Y', 'YES']
    if not should_continue:
        return
    job_group_ids_str = ','.join(str(x) for x in job_group_ids)
    p = subprocess.Popen([ 'scancel', job_group_ids_str ], stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print()
    input('Done. Press any key to continue.')


if __name__ == '__main__':
    setup_environment()
    while True:
        main_menu()
