# -*- coding: utf-8 -*-
"""
Batch audio feature extractor for acoss
"""
import argparse
import time
import glob
import os
import deepdish as dd
#from joblib import Parallel, delayed
import ray
import psutil

from progress.bar import Bar
from shutil import rmtree

from acoss.utils import log, read_txt_file, savelist_to_file, create_audio_path_batches, split_list_with_N_elements
from acoss.features import AudioFeatures

__all__ = ['PROFILE', 
            'compute_features', 
            'compute_features_from_list_file', 
            'batch_feature_extractor']

# PROFILE = {
#            'sample_rate': 44100,
#            'input_audio_format': '.mp3',
#            'downsample_audio': False,
#            'downsample_factor': 2,
#            'endtime': None,
#            'features': ['hpcp',
#                         'key_extractor',
#                         'madmom_features',
#                         'mfcc_htk']
#         }

from PROFILE import PROFILE_11kHz as PROFILE

_LOG_FILE_PATH = "acoss.extractor.log"
_LOG_FILE = log(_LOG_FILE_PATH)
_ERRORS = list()


def compute_features(audio_path, params=PROFILE):
    """
    Compute a list of audio features for a given audio file as per the extractor profile.

    NOTE: Audio files should be structured in a way that each cover song clique has a folder with it's tracks inside to
          have the correct cover label in the resulted feature dictionary.

          eg: ./audio_dir/
                    /cover_clique_label/ (folder name)
                        /audio_file.mp3 (or any other format)

    :param audio_path: path to audio file
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params)

    :return: a python dictionary with all the requested features computed as key, value pairs.
    """
    feature = AudioFeatures(audio_file=audio_path, sample_rate=params['sample_rate'])
    if feature.audio_vector.shape[0] == 0:
        raise IOError("Empty or invalid audio recording file -%s-" % audio_path)

    if params['endtime']:
        feature.audio_vector = feature.audio_slicer(endTime=params['endtime'])
    if params['downsample_audio']:
        feature.audio_vector = feature.resample_audio(params['sample_rate'] / params['downsample_factor'])

    out_dict = dict()
    # now we compute all the listed features in the profile dict and store the results to a output dictionary
    for method in params['features']:
        out_dict[method] = getattr(feature, method)(params=params[method])

    track_id = os.path.basename(audio_path).replace(params['input_audio_format'], '')
    out_dict['track_id'] = track_id

    label = audio_path.split('/')[-2]
    out_dict['label'] = label

    out_dict['params'] = params

    return out_dict

@ray.remote
def compute_features_from_list_file(input_txt_file, feature_dir, params=PROFILE):
    """
    Compute specified audio features for a list of audio file paths and store to disk as .h5 file
    from a given input text file.
    It is a wrapper around 'compute_features'.

    :param input_txt_file: a text file with a list of audio file paths
    :param feature_dir: a path
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params)

    :return: None
    """

    start_time = time.monotonic()
    _LOG_FILE.info("Extracting Batch File: %s " % input_txt_file)
    data = read_txt_file(input_txt_file)
    data = [path for path in data if os.path.exists(path)]
    if len(data) < 1:
        _LOG_FILE.debug("Empty collection txt file -%s- !" % input_txt_file)
        raise IOError("Empty collection txt file -%s- !" % input_txt_file)

    if params['verbose'] >= 1:
        progress_bar = Bar('acoss.extractor.compute_features_from_list_file',
                            max=len(data),
                            suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    for song in data:
        _LOG_FILE.info("Extracting features for %s " % song)
        try:
            work_id = song.split('/')[-2]
            work_dir = os.path.join(feature_dir, work_id+"/")
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            filename = work_dir + os.path.basename(song).replace(params['input_audio_format'], '') + '.h5'

            # extract features
            if params['overwrite'] or not os.path.exists(filename):
                feature_dict = compute_features(audio_path=song, params=params)

                #save as h5
                dd.io.save(filename,feature_dict)
        except:
            _ERRORS.append(input_txt_file)
            _ERRORS.append(song)
            _LOG_FILE.debug("Error: skipping computing features for audio file --%s-- " % song)

        if params['verbose'] >= 1:
            progress_bar.next()

    if params['verbose'] >= 1:
        progress_bar.finish()

    _LOG_FILE.info("Process finished in - %s - seconds" % (start_time - time.time()))


def batch_feature_extractor(dataset_csv, audio_dir, feature_dir, batchesdir,  n_workers=-1, mode='parallel', params=PROFILE):
    """
    Compute parallelised feature extraction process from a collection of input audio file path txt files

    :param
        dataset_csv: dataset csv file
        audio_dir: path where the audio files are stored
        feature_dir: path where the computed audio features should be stored
        n_workers: no. of workers for parallelisation
        mode: whether to run the extractor in 'single' or 'parallel' mode.
        params: profile dict with params

    :return: None
    """
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    batch_file_dir = batchesdir

    create_audio_path_batches(dataset_csv,
                              dir_to_save=batch_file_dir,
                              root_audio_dir=audio_dir,
                              audio_format=params['input_audio_format'],
                              n_workers=n_workers)

    collection_files = glob.glob(batch_file_dir + '*.txt')
    feature_path = [feature_dir for i in range(len(collection_files))]
    param_list = [params for i in range(len(collection_files))]
    args = zip(collection_files, feature_path, param_list)
    _LOG_FILE.info("Computing batch feature extraction using '%s' mode the profile: %s \n" % (mode, params))

    if mode == 'parallel':
        music_groups = split_list_with_N_elements(collection_files,n_workers)

        for cpaths in music_groups:
            params_id = ray.put(params)
            features_dir_id = ray.put(feature_dir)
            ray.get([compute_features_from_list_file.remote(cpath,features_dir_id,params_id) for cpath in cpaths])

        #Parallel(n_jobs=n_workers, verbose=1)(delayed(compute_features_from_list_file)\
        #                                      (cpath, fpath, param) for cpath, fpath, param in args)
    elif mode == 'single':
        tic = time.monotonic()
        #progressbar = Bar('acoss.extractor.batch_feature_extractor',
        #                max=len(args),
        #                suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for cpath, fpath, param in args:
            compute_features_from_list_file(cpath, fpath, param)
        #    progressbar.next()
        #progressbar.finish()
        _LOG_FILE.info("Single mode feature extraction finished in %s" % (time.monotonic() - tic))
    else:
        raise IOError("Wrong value for the parameter 'mode'. Should be either 'single' or 'parallel'")
    savelist_to_file(_ERRORS,'_erros_acoss.extractors.txt')
    rmtree(batch_file_dir)
    _LOG_FILE.info("Log file located at '%s'" % _LOG_FILE_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="With command-line args, it does batch feature extraction of  \
             collection of audio files using multiple threads", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset_csv", action="store",
                        help="path to input dataset csv file")
    parser.add_argument("-a", "--audio_dir", action="store",
                        help="path to the main audio directory of dataset")
    parser.add_argument("-p", "--feature_dir", action="store",
                        help="path to directory where the audio features should be stored")
    parser.add_argument("-f", "--feature_list", action="store", type=str, dest='alist', nargs='*',
                        default="['hpcp', 'key_extractor', 'crema', 'madmom_features', 'mfcc_htk']",
                        help="List of features to compute. Eg. ['hpcp' 'crema']")
    parser.add_argument("-m", "--run_mode", action="store", default='parallel',
                        help="Whether to run the extractor in single or parallel mode. "
                             "Choose one of ['single', 'parallel']")
    parser.add_argument("-n", "--workers", action="store",type=int, default=-1,
                        help="No of workers for running the batch extraction process. Only valid in 'parallel' mode.")
    parser.add_argument("-v", "--verbose", action="store", type=int, default=0,
                        help="It shows the progress bar.")
    parser.add_argument("-w", "--redis_password", action="store", type=str, default="",
                        help="Cluster redisPassword - generated by slurm script.")
    parser.add_argument("-c", "--type_cluster", action="store", type=int, default=0,
                        help="Cluster type to use (0 - without cluster, 1 - Slurm) - default 0.")
    parser.add_argument("-r", "--overwrite", action="store", type=int, default=1,
                        help="Overwrite existing files (0, 1) - default 1.")
    parser.add_argument("-b", "--batchesdir", action="store", type=str, default="./batches/",
                        help="batches dir.")


    cmd_args = parser.parse_args()

    print("Args: %s" % cmd_args)

    if not os.path.exists(cmd_args.feature_dir):
        os.mkdir(cmd_args.feature_dir)

    feature_list = cmd_args.alist
    updated_profile = PROFILE.copy()
    del updated_profile['features']
    updated_profile['features'] = feature_list
    updated_profile['verbose'] = cmd_args.verbose
    updated_profile['overwrite'] = bool(cmd_args.overwrite)

    # Start Ray
    num_cpus = cmd_args.workers
    print("Nr CPUs: %d" % num_cpus)

    if cmd_args.type_cluster == 1: # Slurm Cluster Configuration
        ray.init(address=os.environ["ip_head"],
                 redis_password=cmd_args.redis_password) # 1 GB
    else: # Use Ray in PC
        if cmd_args.workers == -1:
            num_cpus = psutil.cpu_count(logical=False)

        ray.init(num_cpus=num_cpus)

    batch_feature_extractor(dataset_csv=cmd_args.dataset_csv,
                            audio_dir=cmd_args.audio_dir,
                            feature_dir=cmd_args.feature_dir,
                            batchesdir = cmd_args.batchesdir,
                            n_workers=num_cpus,
                            mode=cmd_args.run_mode,
                            params=updated_profile)

    print("... Done ....")
    print(" -- PROFILE INFO -- \n %s" % updated_profile)



    #List File by Datetime
    # find -name '*.h5' -exec ls -rtl {} +