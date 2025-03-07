#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Imports
########################################

import sys, os
# Update this to point to the directory where you copied the SciAnalysis base code
#SciAnalysis_PATH='/home/kyager/current/code/SciAnalysis/main/'
#SciAnalysis_PATH='/nsls2/xf11bm/software/SciAnalysis/'
SciAnalysis_PATH='/nsls2/data/cms/legacy/xf11bm/software/SciAnalysis/'
SciAnalysis_PATH in sys.path or sys.path.append(SciAnalysis_PATH)

import glob
from SciAnalysis import tools
from SciAnalysis.XSAnalysis.Data import *
from SciAnalysis.XSAnalysis import Protocols

import time


# Define some custom analysis routines
########################################
from SciAnalysis.Result import * # The Results() class allows one to extract data from XML files.


def autonomous_result(xml_file, clean=True, verbosity=3):
    # DEPRECATED: Use SQL database instead:
    #new_result = ResultsDB().extract_single(infile)
    
    time.sleep(0.5) # Kludge to avoid corrupting XML file?
    
    extractions = [ 
        [ 'metadata_extract', 
            [
            'x_position', 
            'y_position',
            #'sequence_ID',
            ] 
        ],
        ['circular_average_q2I_fit', 
            [
            'fit_peaks_prefactor1', 
            'fit_peaks_x_center1', 
            'fit_peaks_sigma1', 
            'fit_peaks_chi_squared', 
            'fit_peaks_prefactor1_error', 
            'fit_peaks_x_center1_error', 
            'fit_peaks_sigma1_error', 
            ] 
        ],
        #['linecut_angle_fit', 
            #[
            #'fit_eta_eta', 
            #'orientation_factor', 
            #'orientation_angle', 
            #'fit_eta_span_prefactor',
            #] 
        #],
        ]


    results_dict = Results().extract_dict([xml_file], extractions, verbosity=verbosity)[0]

    if clean:
        for key, result in results_dict.items():
            if key!='filename':
                results_dict[key] = np.nan_to_num(float(result))
        

    return results_dict     


# Experimental parameters
########################################

calibration = Calibration(wavelength_A=0.9184) # 13.5 keV
calibration.set_image_size(1475, height=1679) # Pilatus2M
calibration.set_pixel_size(pixel_size_um=172.0)
calibration.set_beam_position(754, 1075)

calibration.set_distance(5.03) # 5m


mask_dir = SciAnalysis_PATH + '/SciAnalysis/XSAnalysis/masks/'
mask = Mask(mask_dir+'Dectris/Pilatus2M_gaps-mask.png')
mask.load('./mask.png')



# Files to analyze
########################################
source_dir = '../raw/'
output_dir = './'

pattern = '*'

infiles = glob.glob(os.path.join(source_dir, pattern+'.tiff'))
infiles.sort()


# Analysis to perform
########################################

load_args = { 'calibration' : calibration, 
             'mask' : mask,
             #'background' : source_dir+'empty*saxs.tiff',
             #'transmission_int': '../../data/Transmission_output.csv', # Can also specify an float value.
             }
run_args = { 'verbosity' : 3,
            #'save_results' : ['xml', 'plots', 'txt', 'hdf5'],
            }

process = Protocols.ProcessorXS(load_args=load_args, run_args=run_args)
#process.connect_databroker('cms') # Access databroker metadata


patterns = [
            ['theta', '.+_th(\d+\.\d+)_.+'] ,
            ['x_position', '.+_x(-?\d+\.\d+)_.+'] ,
            ['y_position', '.+_yy(-?\d+\.\d+)_.+'] ,
            #['anneal_time', '.+_anneal(\d+)_.+'] ,
            #['cost', '.+_Cost(\d+\.\d+)_.+'] ,
            ['annealing_temperature', '.+_T(\d+\.\d\d\d)C_.+'] ,
            #['annealing_time', '.+_(\d+\.\d)s_T.+'] ,
            #['annealing_temperature', '.+_localT(\d+\.\d)_.+'] ,
            #['annealing_time', '.+_clock(\d+\.\d\d)_.+'] ,
            #['o_position', '.+_opos(\d+\.\d+)_.+'] ,
            #['l_position', '.+_lpos(\d+\.\d+)_.+'] ,
            ['exposure_time', '.+_(\d+\.\d+)s_\d+_saxs.+'] ,
            ['sequence_ID', '.+_(\d+).+'] ,
            ]

protocols = [
    #Protocols.HDF5(save_results=['hdf5'])
    #Protocols.calibration_check(show=False, AgBH=True, q0=0.010, num_rings=4, ztrim=[0.05, 0.05], ) ,
    #Protocols.circular_average(ylog=True, plot_range=[0, 0.12, None, None], label_filename=True) ,
    #Protocols.thumbnails(crop=None, resize=1.0, blur=None, cmap=cmap_vge, ztrim=[0.01, 0.001]) ,
    
    #Protocols.circular_average_q2I_fit(show=False, q0=0.0140, qn_power=2.5, sigma=0.0008, plot_range=[0, 0.06, 0, None], fit_range=[0.008, 0.022]) ,
    #Protocols.circular_average_q2I_fit(qn_power=3.5, trim_range=[0.005, 0.03], fit_range=[0.007, 0.019], q0=0.0120, sigma=0.0008) ,
    Protocols.circular_average_q2I_fit(qn_power=3.0, trim_range=[0.005, 0.035], fit_range=[0.008, 0.03], q0=0.0180, sigma=0.001) ,
    
    #Protocols.databroker_extract(constraints={'measure_type':'measure'}, timestamp=True, sectino='start'),
    Protocols.metadata_extract(patterns=patterns) ,
    ]
    



# Helpers
########################################
# DEPRECATED in favor of tools.val_stats
def print_d(d, i=4):
    '''Simple helper to print a dictionary.'''
    for k, v in d.items():
        if isinstance(v,dict):
            print('{}{} : <dict>'.format(' '*i,k))
            print_d(v, i=i+4)
        elif isinstance(v,(np.ndarray)):
            print('{}{} : Ar{}: {}'.format(' '*i,k,v.shape,v))
        elif isinstance(v,(list,tuple)):
            print('{}{} : L{}: {}'.format(' '*i,k,len(v),v))
        else:
            print('{}{} : {}'.format(' '*i,k,v))

def print_results(results):
    '''Simple helper to print out a list of dictionaries.'''
    for i, result in enumerate(results):
        print(i)
        print_d(result)

def print_n(d):
    '''Simple helper to print nested arrays/dicts'''
    if isinstance(d, (list,tuple,np.ndarray)):
        print_results(d)
    elif isinstance(d, dict):
        print_d(d)
    else:
        print(d)

def val_stats(values, name='z'):
    span = np.max(values)-np.min(values)
    print("  {} = {:.2g} ± {:.2g} (span {:.2g}, from {:.3g} to {:.3g})".format(name, np.average(values), np.std(values), span, np.min(values), np.max(values)))


# Inspect .npy files
########################################
# This code can be pasted into a new ipython shell
# to help inspect the .npy files that are passed
# in the AE loop.
'''

import numpy as np

def print_d(d, i=4):
    for k, v in d.items():
        if isinstance(v,dict):
            print('{}{} : <dict>'.format(' '*i,k))
            print_d(v, i=i+4)
        else:
            print('{}{} : {}'.format(' '*i,k,v))

def print_results(results):
    for i, result in enumerate(results):
        print(i)
        print_d(result)

        
results = np.load('analyze-received.npy', allow_pickle=True); print_results(results)
results = np.load('analyze-sent.npy', allow_pickle=True); print_results(results)

results = np.load('../../measure-received.npy', allow_pickle=True); print_results(results)
results = np.load('../../measure-sent.npy', allow_pickle=True); print_results(results)

results = np.load('../../gpcam/scripts/decision-received.npy', allow_pickle=True); print_results(results)
results = np.load('../../gpcam/scripts/decision-sent.npy', allow_pickle=True); print_results(results)

#print_results(results)

'''

def determine_infile(filename, source_dir='./', suffix='_saxs.tiff', filename_re=None, verbosity=3):
    #  Usage: 
    # filename_re = re.compile('.+_x(-?\d+\.\d+)_y(-?\d+\.\d+)_.+_(\d+)_saxs.+') # TOCHANGE
    # determine_infile(result['filename'], source_dir=source_dir, filename_re=filename_re, verbosity=verbosity, suffix='_saxs.tiff') # TOCHANGE

    infile = '{}{}{}'.format(source_dir, filename, suffix)

    # Code to handle bug where saved filename doesn't exactly match what is specified in metadata:
    if not os.path.exists(infile):
        if verbosity>=1:
            print('Specified infile is missing. We will attempt to locate the right file based on sequence_ID.')
        if verbosity>=5:
            print('  infile: {}'.format(infile))
            
        if filename_re is None:
            if verbosity>=1:
                print("    No RE provided. Aborting.")
            return None
        
        else:
            m = filename_re.match(infile)
            if m:
                sID = int(m.groups()[2])
                if verbosity>=2:
                    print('    sequence ID: {:d}'.format(sID))
                mfiles = glob.glob('{}*_{:d}_saxs.tiff'.format(source_dir, sID)) # TOCHANGE
                if len(mfiles)<1:
                    if verbosity>=1:
                        print('    No file matches sequence ID {}.'.format(sID))
                elif len(mfiles)==1:
                    infile = mfiles[0]
                    if verbosity>=1:
                        print('    Using filename: {}'.format(infile))
                else:
                    if verbosity>=1:
                        print('    {} files match sequence ID {}'.format(len(mfiles), sID))
                        print('    Aborting.')
                    return None
            else:
                if verbosity>=1:
                    print("    RE did not match. Aborting.")
                return None
    
    return infile



# Run autonomous loop
########################################
def run_autonomous_loop(protocols, clear=False, force_load=False, republish=False, verbosity=3, simulate=False):
    
    # IMPORTANT NOTE: Search for "# TOCHANGE" in the code below for
    # beamline-specific and experiment-specific assumptions that need
    # to be adjusted.


    # Connect to queue to receive the next analysis command
    #code_PATH='/nsls2/xf12id2/data/CFN/2020_3/MFukuto/'
    code_PATH='../../../'
    code_PATH in sys.path or sys.path.append(code_PATH)

    #from CustomQueue import Queue_analyze as queue
    from CustomS3 import Queue_analyze as queue
    q = queue()

    if clear:
        q.clear()
    if republish:
        q.republish()
    
    
    if verbosity>=3:
        print('\n\n\n')
        print('=============================')
        print('==  Autonomous Experiment  ==')
        print('=============================')
        print('\n')
        
        
    filename_re = re.compile('.+_x(-?\d+\.\d+)_y(-?\d+\.\d+)_.+_(\d+)_saxs.+') # TOCHANGE
    
    while True: # Loop forever
        
        results = q.get(force_load=force_load) # Get analysis command
        force_load = False # Only force a reload on the 1st iteration
        
        num_to_analyze = int( sum( 1.0 for result in results if result['analyzed'] is False ) )
        
        if verbosity>=3:
            print('Analysis requested for {} results (total {} results)'.format(num_to_analyze, len(results)))
        if verbosity>=10:
            tools.print_results(results)
        
        ianalyze = 0
        for i, result in enumerate(results):
            
            if 'analyzed' in result and result['analyzed'] is False and 'filename' in result:
                ianalyze += 1
                
                infile = determine_infile(result['filename'], source_dir=source_dir, filename_re=filename_re, verbosity=verbosity, suffix='_saxs.tiff') # TOCHANGE

                if verbosity>=3:
                    print('        Analysis for result {}/{}, file: {}'.format(ianalyze, num_to_analyze, infile))


                if simulate:
                    value, variance = np.random.random()*10, 1.0

                else:
                    process.run([infile], protocols, output_dir=output_dir, force=True)
                    
                    if False:
                        # Get the result of this analysis using XML files
                        xml_file = '{}{}{}'.format( './results/', infile[len(source_dir):-5], '.xml' ) # TOCHANGE
                        new_result = autonomous_result(xml_file)
                        #result.update(new_result)
                        if 'metadata' not in result:
                            result['metadata'] = None
                        if result['metadata'] is None:
                            result['metadata'] = { 'SciAnalysis': new_result }
                        else:
                            result['metadata'].update({ 'SciAnalysis': new_result })

                        # TOCHANGE
                        #value = new_result['circular_average_q2I_fit__fit_peaks_prefactor1']
                        #variance = np.square(new_result['circular_average_q2I_fit__fit_peaks_prefactor1_error'])
                        value = new_result['circular_average_q2I_fit__fit_peaks_chi_squared']*1e9
                        variance = 0
                        
                    if False:
                        # Get information about this measurement from Bluesky/databroker
                        
                        # If Protocols.databroker_extract was run, then you can:
                        #results_dict = ResultsDB().extract_single(infile, verbosity=verbosity)
                        #value = new_results['databroker_extract']['sample_phi']
                        #value = new_results['databroker_extract']['start']['motor_SAXSx']
                        
                        # If process.connect_databroker('cms') was activated, then you can:
                        h = process.get_db(uid=result['uid']) # uid matching is most robust
                        #filename = tools.Path(infile).stem[:-5] # remove trailing "_saxs"
                        #scan_id = int(filename.split('_')[-1])
                        #h = process.get_db(filename=filename, scan_id=scan_id, recent_days=7)
                        
                        value = h['start']['motor_SAXSx']
                        
                        
                    if True:
                        # Get the result of this analysis from the SQL database
                        results_dict = ResultsDB().extract_single(infile, verbosity=verbosity)
                        if 'metadata' not in result:
                            result['metadata'] = None
                        if result['metadata'] is None:
                            result['metadata'] = { 'SciAnalysis': results_dict }
                        else:
                            result['metadata'].update({ 'SciAnalysis': results_dict })

                        # TOCHANGE
                        value = results_dict['circular_average_q2I_fit']['fit_peaks_prefactor1']
                        error = results_dict['circular_average_q2I_fit']['fit_peaks_prefactor1_error']
                        variance = np.square(error)


                # Package for gpCAM
                if verbosity>=5:
                    print('Packaging result: {:.4g} ± {:.4g}'.format(value, variance))

                n = 1.0e0 # TOCHANGE Rescale values for gpCAM, so that they are roughly of order unity (this avoids machine precision problems)
                result['value'] = value*n
                result['variance'] = variance*n
                result['analyzed'] = True


        if verbosity>=3:
            print('Analyzed {} results'.format(ianalyze))
        if verbosity>=1 and ianalyze<1:
            print('WARNING: No results were analyzed.')
        if verbosity>=5:
            print_results(results)
        
        q.publish(results)
    

#process.run(infiles, protocols, output_dir=output_dir, force=True)
run_autonomous_loop(protocols, clear=False, force_load=False, republish=False, verbosity=3)

