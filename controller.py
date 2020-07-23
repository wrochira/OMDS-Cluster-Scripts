"""
Notes:
* Before running this script on Viking, be sure to load an appropriate Python 3 module,
  e.g. with the command `module load lang/SciPy-bundle/2020.03-foss-2019b-Python-3.7.4`
"""

import os
import time
import shutil
import datetime
import subprocess


# Globals
### Constants
PARAMETERS = { 'CPU_THREADS' : { 'constant' : True }, 'SHOW_MSD' : { 'constant' : True }, 'SHOW_DIFFUSION' : { 'constant' : True }, 'SHOW_MEMBRANE' : { 'constant' : True }, 'SHOW_REFERENCE_CIRCLE' : { 'constant' : True }, 'SHOW_LOCAL_REGION' : { 'constant' : True }, 'LOCAL_RANGE' : { 'constant' : True }, 'DISPLAY_VISUALISATION' : { 'constant' : True }, 'MAGNIFICATION' : { 'constant' : True }, 'SAVE_VISUALISATION' : { 'constant' : True }, 'SAVE_FINAL_STATE_ONLY' : { 'constant' : True }, 'DATA_INTERVAL' : { 'constant' : True }, 'EXPORT_XML' : { 'constant' : True }, 'EXPORT_XLSX' : { 'constant' : True }, 'EXPORT_TRAJECTORIES' : { 'constant' : True }, 'TRAJECTORY_INTERVAL' : { 'constant' : True }, 'N_PROTEINS' : { 'constant' : False }, 'N_STEPS' : { 'constant' : True }, 'TIME_STEP' : { 'constant' : True }, 'CENTRE_WEIGHTING' : { 'constant' : True }, 'FIELD_DEPTH' : { 'constant' : True }, 'INTENSITY_0' : { 'constant' : True }, 'LAMBDA_0' : { 'constant' : True }, 'TIRF_ANGLE' : { 'constant' : True }, 'N1' : { 'constant' : True }, 'N2' : { 'constant' : True }, 'TEMPERATURE' : { 'constant' : True }, 'VISCOSITY' : { 'constant' : True }, 'RADIUS' : { 'constant' : True }, 'LENGTH' : { 'constant' : True }, 'XMIN' : { 'constant' : True }, 'XMAX' : { 'constant' : True }, 'YMIN' : { 'constant' : True }, 'YMAX' : { 'constant' : True }, 'PERCENTAGE_OCCUPANCY' : { 'constant' : False }, 'DIFFUSION_RANGE' : { 'constant' : False }, 'MEMBRANE_DIFFUSING' : { 'constant' : True }, 'MEMBRANE_ROTATING' : { 'constant' : True }, 'TARGET_DIFFUSING' : { 'constant' : True }, 'TARGET_ROTATING' : { 'constant' : True }, 'TARGET_INTERACTING' : { 'constant' : True }, 'ALIGNED_DISSOCIATION' : { 'constant' : True }, 'TARGET_KOFF' : { 'constant' : False }, 'STICKY_PROTEINS' : { 'constant' : True }, 'R_BTUB' : { 'constant' : True }, 'HEIGHT_BTUB' : { 'constant' : True }, 'D_LAT_BTUB' : { 'constant' : False }, 'D_LAT_ASS_BTUB' : { 'constant' : False }, 'RELPROP_BTUB' : { 'constant' : False }, 'STARTING_PATCH_COUNT_BTUB' : { 'constant' : False }, 'STARTING_PATCH_ANGLE_BTUB' : { 'constant' : False }, 'COLOUR_BTUB' : { 'constant' : True }, 'R_OMPF' : { 'constant' : True }, 'HEIGHT_OMPF' : { 'constant' : True }, 'D_LAT_OMPF' : { 'constant' : False }, 'D_LAT_ASS_OMPF' : { 'constant' : False }, 'RELPROP_OMPF' : { 'constant' : False }, 'STARTING_PATCH_COUNT_OMPF' : { 'constant' : False }, 'STARTING_PATCH_ANGLE_OMPF' : { 'constant' : False }, 'COLOUR_OMPF' : { 'constant' : True }, 'R_OMPA' : { 'constant' : True }, 'HEIGHT_OMPA' : { 'constant' : True }, 'D_LAT_OMPA' : { 'constant' : False }, 'D_LAT_ASS_OMPA' : { 'constant' : False }, 'RELPROP_OMPA' : { 'constant' : False }, 'STARTING_PATCH_COUNT_OMPA' : { 'constant' : False }, 'STARTING_PATCH_ANGLE_OMPA' : { 'constant' : False }, 'COLOUR_OMPA' : { 'constant' : True }, 'R_TOLA' : { 'constant' : True }, 'HEIGHT_TOLA' : { 'constant' : True }, 'D_LAT_TOLA' : { 'constant' : False }, 'D_LAT_ASS_TOLA' : { 'constant' : False }, 'RELPROP_TOLA' : { 'constant' : False }, 'STARTING_PATCH_COUNT_TOLA' : { 'constant' : False }, 'STARTING_PATCH_ANGLE_TOLA' : { 'constant' : False }, 'COLOUR_TOLA' : { 'constant' : True }, 'COLOUR_DIFF' : { 'constant' : True }, 'SIMULATE_INNER_MEMBRANE' : { 'constant' : True }, 'IM_PERCENTAGE_OCCUPANCY' : { 'constant' : False } }
BINARIES_DIR = './binaries/'
PARAMSETS_DIR = './parameter_sets/'
RIP_DIR = './runs_in_progress/'
RS_DIR = './runs_saved/'
### Variables
DEFAULT_PS = { }
CHOSEN_PS = { }
DISCREPANCIES = [ ] # [ ('Variable name', ('Variable changes')), ... ]
CURRENT_DIR = ''
RUN_NAME = ''
NUM_INSTANCES = -1
NUM_PARAMSETS = -1


def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
    print()


def continue_prompt():
    print()
    print('Continue?')
    choice = input('> ')
    if choice.upper() not in ['Y', 'YES']:
        exit()


def setup_environment():
    for dir_name in (BINARIES_DIR, PARAMSETS_DIR, RIP_DIR, RS_DIR):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)


def get_purpose():
    clear_screen()
    print('Select function:')
    print('1) Start new job set.')
    print('2) Status update.')
    return int(input('> '))


def load_ps(ps_type='Choice'):
    clear_screen()
    if ps_type == 'Default':
        ps_filenames = sorted([ x for x in os.listdir(PARAMSETS_DIR) if 'default' in x and x[-4:] == '.xml' ])
        chosen_filename = ps_filenames[-1]
    elif ps_type == 'Choice':
        ps_filenames = sorted([ x for x in os.listdir(PARAMSETS_DIR) if 'default' not in x and x[-4:] == '.xml' ])
        print('Choose parameter set:')
        for i in range(len(ps_filenames)):
            print(str(i+1) + ')', ps_filenames[i])
        choice = int(input('> '))
        print()
        chosen_filename = ps_filenames[choice-1]

    parameter_set = { }
    with open(os.path.join(PARAMSETS_DIR, chosen_filename), 'r') as infile:
        for line in infile.readlines():
            for parameter_name in PARAMETERS.keys():
                if '"' + parameter_name + '"' in line:
                    parameter_value = line.split('VALUE="')[-1].split('"')[0]
                    if ps_type == 'Default':
                        parameter_set[parameter_name] = parameter_value
                    else:
                        if parameter_name in parameter_set:
                            parameter_set[parameter_name].append(parameter_value)
                        else:
                            parameter_set[parameter_name] = [ parameter_value ]
    return parameter_set


def review_ps():
    global CHOSEN_PS
    global DISCREPANCIES
    global NUM_INSTANCES
    global NUM_PARAMSETS
    
    if sorted(CHOSEN_PS.keys()) != sorted(PARAMETERS.keys()):
        print('Parameter names mismatch!')
        return False

    lengths = [ ]
    for parameter_name in PARAMETERS.keys():
        lengths.append(len(CHOSEN_PS[parameter_name]))
    if lengths.count(lengths[0]) != len(lengths):
        print('Parameter lengths mismatch!')
        return False
    NUM_PARAMSETS = lengths[0]

    mismatch_constants = [ ]
    mismatch_variables = [ ]
    for parameter_name in PARAMETERS.keys():
        default_value = DEFAULT_PS[parameter_name]
        chosen_values = CHOSEN_PS[parameter_name]
        if default_value != chosen_values[0] or chosen_values.count(chosen_values[0]) != len(chosen_values):
            if PARAMETERS[parameter_name]['constant']:
                mismatch_constants.append((parameter_name, chosen_values, default_value))
            elif parameter_name != 'N_PROTEINS':
                mismatch_variables.append((parameter_name, chosen_values, default_value))
    
    clear_screen()
    if len(mismatch_constants) > 0:
        print('Discrepancies in constants:')
        for i in range(len(mismatch_constants)):
            mm = mismatch_constants[i]
            DISCREPANCIES.append( (mm[0], mm[1]) )
            if mm[1].count(mm[1][0]) == len(mm[1]):
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], 'All ' + mm[1][0], '(def. ' + mm[2] + ')')
            else:
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], str(mm[1]), '(def. ' + mm[2] + ')')
        print()
        print('THESE WILL ALL BE LEFT AS THEY ARE.')
    else:
        print('No discrepancies in constants.')
    continue_prompt()

    clear_screen()
    if len(mismatch_variables) > 0:
        print('Discrepancies in variables:')
        for i in range(len(mismatch_variables)):
            mm = mismatch_variables[i]
            DISCREPANCIES.append( (mm[0], mm[1]) )
            if mm[1].count(mm[1][0]) == len(mm[1]):
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], 'All ' + mm[1][0], '(def. ' + mm[2] + ')')
            else:
                print(str(i+1) + ')', (mm[0] + ' ' * 30)[:30], str(mm[1]), '(def. ' + mm[2] + ')')
        print()
        print('THESE WILL ALL BE LEFT AS THEY ARE.')
    else:
        print('No discrepancies in variables.')
    continue_prompt()

    clear_screen()
    print('How many instances per parameter set?')
    NUM_INSTANCES = int(input('> '))

    clear_screen()
    print('How many runs per instance?')
    instance_runs = int(input('> '))
    total_runs = NUM_INSTANCES * instance_runs

    clear_screen()
    print('Total number of runs will be', str(total_runs) + ',', 'split over', NUM_INSTANCES, 'instance(s) per parameter set.')
    continue_prompt()

    CHOSEN_PS['N_PROTEINS'] = [ str(instance_runs) for value in CHOSEN_PS['N_PROTEINS'] ]
    for mm in mismatch_constants:
        parameter_name = mm[0]
        CHOSEN_PS[parameter_name] = [ DEFAULT_PS[parameter_name] for value in CHOSEN_PS[parameter_name] ]
    for mm in mismatch_variables:
        pass # TODO: Do stuff here once editing has been implemented


def get_latest_binary():
    global LATEST_BINARY
    binaries = sorted([ x for x in os.listdir(BINARIES_DIR) if x[-4:] == '.jar' ])
    LATEST_BINARY = binaries[-1]
    shutil.copy2(os.path.join(BINARIES_DIR, LATEST_BINARY), CURRENT_DIR)


def generate_paramset():
    with open(os.path.join(CURRENT_DIR, RUN_NAME + '.xml'), 'w') as outfile:
        outfile.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        outfile.write('<EXPERIMENT>\n')
        for i in range(len(CHOSEN_PS['CPU_THREADS'])):
            outfile.write('\t<PARAMETERS>\n')
            outfile.write('\t\t<PARAM NAME="CPU_THREADS" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['CPU_THREADS'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SHOW_MSD" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SHOW_MSD'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SHOW_DIFFUSION" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SHOW_DIFFUSION'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SHOW_MEMBRANE" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SHOW_MEMBRANE'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SHOW_REFERENCE_CIRCLE" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SHOW_REFERENCE_CIRCLE'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SHOW_LOCAL_REGION" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SHOW_LOCAL_REGION'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="LOCAL_RANGE" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['LOCAL_RANGE'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="DISPLAY_VISUALISATION" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['DISPLAY_VISUALISATION'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="MAGNIFICATION" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['MAGNIFICATION'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SAVE_VISUALISATION" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SAVE_VISUALISATION'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SAVE_FINAL_STATE_ONLY" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SAVE_FINAL_STATE_ONLY'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="DATA_INTERVAL" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['DATA_INTERVAL'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="EXPORT_XML" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['EXPORT_XML'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="EXPORT_XLSX" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['EXPORT_XLSX'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="EXPORT_TRAJECTORIES" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['EXPORT_TRAJECTORIES'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TRAJECTORY_INTERVAL" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['TRAJECTORY_INTERVAL'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="N_PROTEINS" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['N_PROTEINS'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="N_STEPS" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['N_STEPS'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TIME_STEP" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['TIME_STEP'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="CENTRE_WEIGHTING" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['CENTRE_WEIGHTING'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="FIELD_DEPTH" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['FIELD_DEPTH'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="INTENSITY_0" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['INTENSITY_0'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="LAMBDA_0" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['LAMBDA_0'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TIRF_ANGLE" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['TIRF_ANGLE'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="N1" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['N1'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="N2" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['N2'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TEMPERATURE" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['TEMPERATURE'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="VISCOSITY" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['VISCOSITY'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="RADIUS" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['RADIUS'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="LENGTH" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['LENGTH'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="XMIN" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['XMIN'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="XMAX" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['XMAX'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="YMIN" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['YMIN'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="YMAX" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['YMAX'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="PERCENTAGE_OCCUPANCY" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['PERCENTAGE_OCCUPANCY'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="DIFFUSION_RANGE" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['DIFFUSION_RANGE'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="MEMBRANE_DIFFUSING" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['MEMBRANE_DIFFUSING'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="MEMBRANE_ROTATING" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['MEMBRANE_ROTATING'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TARGET_DIFFUSING" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['TARGET_DIFFUSING'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TARGET_ROTATING" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['TARGET_ROTATING'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TARGET_INTERACTING" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['TARGET_INTERACTING'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="ALIGNED_DISSOCIATION" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['ALIGNED_DISSOCIATION'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="TARGET_KOFF" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['TARGET_KOFF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STICKY_PROTEINS" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['STICKY_PROTEINS'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="R_BTUB" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['R_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="HEIGHT_BTUB" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['HEIGHT_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_BTUB" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_ASS_BTUB" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_ASS_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="RELPROP_BTUB" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['RELPROP_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_COUNT_BTUB" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['STARTING_PATCH_COUNT_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_ANGLE_BTUB" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['STARTING_PATCH_ANGLE_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="COLOUR_BTUB" TYPE="java.awt.Color" VALUE="' + CHOSEN_PS['COLOUR_BTUB'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="R_OMPF" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['R_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="HEIGHT_OMPF" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['HEIGHT_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_OMPF" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_ASS_OMPF" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_ASS_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="RELPROP_OMPF" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['RELPROP_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_COUNT_OMPF" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['STARTING_PATCH_COUNT_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_ANGLE_OMPF" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['STARTING_PATCH_ANGLE_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="COLOUR_OMPF" TYPE="java.awt.Color" VALUE="' + CHOSEN_PS['COLOUR_OMPF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="R_OMPA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['R_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="HEIGHT_OMPA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['HEIGHT_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_OMPA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_ASS_OMPA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_ASS_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="RELPROP_OMPA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['RELPROP_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_COUNT_OMPA" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['STARTING_PATCH_COUNT_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_ANGLE_OMPA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['STARTING_PATCH_ANGLE_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="COLOUR_OMPA" TYPE="java.awt.Color" VALUE="' + CHOSEN_PS['COLOUR_OMPA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="R_TOLA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['R_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="HEIGHT_TOLA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['HEIGHT_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_TOLA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="D_LAT_ASS_TOLA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['D_LAT_ASS_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="RELPROP_TOLA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['RELPROP_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_COUNT_TOLA" TYPE="java.lang.Integer" VALUE="' + CHOSEN_PS['STARTING_PATCH_COUNT_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="STARTING_PATCH_ANGLE_TOLA" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['STARTING_PATCH_ANGLE_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="COLOUR_TOLA" TYPE="java.awt.Color" VALUE="' + CHOSEN_PS['COLOUR_TOLA'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="COLOUR_DIFF" TYPE="java.awt.Color" VALUE="' + CHOSEN_PS['COLOUR_DIFF'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="SIMULATE_INNER_MEMBRANE" TYPE="java.lang.Boolean" VALUE="' + CHOSEN_PS['SIMULATE_INNER_MEMBRANE'][i] + '" />\n')
            outfile.write('\t\t<PARAM NAME="IM_PERCENTAGE_OCCUPANCY" TYPE="java.lang.Double" VALUE="' + CHOSEN_PS['IM_PERCENTAGE_OCCUPANCY'][i] + '" />\n')
            outfile.write('\t</PARAMETERS>\n')
        outfile.write('</EXPERIMENT>\n')


def generate_jobscript():
    clear_screen()
    print('Run time limit (hh:mm:ss)')
    run_time = input('> ')

    clear_screen()
    print('Email alerts?')
    alerts = input('> ').upper() in ['Y', 'YES']

    if alerts:
        print('Email address')
        email = input('> ')

    with open(os.path.join(CURRENT_DIR, 'jobscript.sh'), 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('#SBATCH --job-name=omd_sim\n')
        if alerts:
            outfile.write('#SBATCH --mail-type=ALL\n')
            outfile.write('#SBATCH --mail-user=' + email + '\n')
        outfile.write('#SBATCH --ntasks=8\n')
        outfile.write('#SBATCH --mem=4gb\n')
        outfile.write('#SBATCH --time=' + run_time + '\n')
        outfile.write('#SBATCH --output=./output_std/omds%j.log\n')
        outfile.write('#SBATCH --error=./output_std/omds%j.err\n')
        outfile.write('#SBATCH --account=biol-stdbom-2019\n')
        outfile.write('#SBATCH --array=0-' + str(NUM_INSTANCES-1) + '\n')
        outfile.write('module load lang/Java/1.8.0_212\n')
        outfile.write('export MALLOC_ARENA_MAX=8\n')
        outfile.write('vmArgs="-Xmx1G -XX:ParallelGCThreads=1 -jar"\n')
        outfile.write('java $vmArgs ./' + LATEST_BINARY + ' ' + RUN_NAME + '.xml ./output_files $PSET_ID $SLURM_ARRAY_TASK_ID')


def generate_launcher():
    with open(os.path.join(CURRENT_DIR, 'launcher.sh'), 'w') as outfile:
        outfile.write('cd "${0%/*}"\n') # Changes to script directory to work from
        for p in range(NUM_PARAMSETS):
            outfile.write('sbatch --export=PSET_ID=' + str(p) + ' jobscript.sh\n')


def generate_variables():
    with open(os.path.join(CURRENT_DIR, 'variables.txt'), 'w') as outfile:
        for discrepancy in DISCREPANCIES:
            outfile.write(discrepancy[0] + ': ' + ', '.join(discrepancy[1]) + '\n')


def generate_run_env():
    global CURRENT_DIR
    global RUN_NAME
    RUN_NAME = datetime.datetime.now().strftime('%y%m%d%H%M')
    CURRENT_DIR = os.path.join(RIP_DIR, RUN_NAME)
    os.mkdir(CURRENT_DIR)
    os.mkdir(os.path.join(CURRENT_DIR, 'output_files/'))
    os.mkdir(os.path.join(CURRENT_DIR, 'output_std/'))
    get_latest_binary()
    generate_paramset()
    generate_jobscript()
    generate_launcher()
    generate_variables()


def submit_and_exit():
    clear_screen()
    print('Ready to launch.')
    input('> ')
    clear_screen()
    print(CURRENT_DIR)
    p = subprocess.Popen(['sh', CURRENT_DIR + '/launcher.sh'], stdout=subprocess.PIPE)

    output_bytes = b''
    while True:
        time.sleep(0.2)
        chunk = p.stdout.read()
        if len(chunk) > 0:
            output_bytes += chunk
        else:
            break

    output = output_bytes.decode()
    print(output)
    job_ids = [ int(x.strip()) for x in output.split('Submitted batch job ') if len(x) > 0 ]

    if len(job_ids) > 0:
        with open(os.path.join(CURRENT_DIR, 'job_ids.txt'), 'w') as outfile:
            outfile.write('Job ID\n')
            for job_id in job_ids:
                outfile.write(str(job_id) + '\n')
        print('\nDone.')
        time.sleep(1)
        status()
    else:
        print('\nLaunch failed.')
        print()
        exit()


def status():
    global CURRENT_DIR

    clear_screen()
    if not CURRENT_DIR:
        runs = sorted(next(os.walk(RIP_DIR))[1])
        print('Available runs:')
        for i in range(len(runs)):
            print(str(i+1) + ')', runs[i])
        print()
        chosen_dir = ''
        while not os.path.isdir(chosen_dir):
            try:
                choice = int(input('> '))
                chosen_dir = os.path.join(RIP_DIR,  runs[choice-1])
            except:
                chosen_dir = ''
        print()
        CURRENT_DIR = chosen_dir

    with open(os.path.join(CURRENT_DIR, 'job_ids.txt'), 'r') as infile:
        infile.readline() # Skip header line
        job_ids = [ int(x.strip()) for x in infile.readlines() ]

    while True:
        job_status_counts = { }
        for job_id in job_ids:
            job_status_counts[job_id] = [ 0, 0, 0, 0 ]
        job_ids_str = ','.join(str(x) for x in job_ids)
        p = subprocess.Popen([ 'sacct', '-j', job_ids_str, '-o', 'jobid,state', '--noheader' ], stdout=subprocess.PIPE)
        time.sleep(1)
        o = p.stdout.read().decode()
        if len(o) == 0:
            time.sleep(1)
            continue
        clear_screen()
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        print(CURRENT_DIR)
        print()
        for line in o.strip().split('\n'):
            if len(line) == 0 or '.' in line:
                continue
            job_info = line[:12].strip().split('_')
            job_id = int(job_info[0])
            run_id = job_info[1]
            state = line[13:].strip()
            if '[' in run_id:
                job_status_counts[job_id][0] += 1
            elif state == 'RUNNING':
                job_status_counts[job_id][1] += 1
            elif state == 'COMPLETED':
                job_status_counts[job_id][2] += 1
            else:
                job_status_counts[job_id][3] += 1

        still_pending = True if job_status_counts[job_id][0] > 0 else False

        print('Job ID        Pending?      # Running     # Completed   # Failed      ')
        for job_id in sorted(job_status_counts.keys()):
            out_line = ''.join([ (str(x) + ' '*14)[:14] for x in [ job_id, still_pending ] + job_status_counts[job_id][1:] ])
            print(out_line)


if __name__ == '__main__':
    setup_environment()
    purpose = get_purpose()
    if purpose == 1:
        DEFAULT_PS = load_ps(ps_type='Default')
        CHOSEN_PS = load_ps(ps_type='Choice')
        review_ps()
        generate_run_env()
        submit_and_exit()
    elif purpose == 2:
        status()
