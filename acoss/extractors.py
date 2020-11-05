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
import sys

from acoss.utils import log, read_txt_file, savelist_to_file, create_audio_path_batches, split_list_with_N_elements
from acoss.features import AudioFeatures

__all__ = ['PROFILE', 
            'compute_features_vector',
            'compute_features', 
            'compute_features_from_list_file', 
            'batch_feature_extractor']
"""
PROFILE = {
           'sample_rate': 44100,
           'hop_length': 512,
           'input_audio_format': '.mp3',
           'downsample_audio': False,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['hpcp',
                        'key_extractor',
                        'madmom_features',
                        'mfcc_htk']
        }
"""
#A linha abiaxo é valida apenas para o prjeto ECAD
from acoss.PROFILE import PROFILE_11kHz as PROFILE

_LOG_FILE_PATH = "acoss.extractor.log"
_LOG_FILE = log(_LOG_FILE_PATH)
_ERRORS = list()

def compute_features_vector(audio_vector, track_id, label, params):
    """
    Created by Atila Xavier
    Compute a list of audio features for a given audio vector as per the extractor profile.

    :param audio_vector: numpy.ndarray with audio
    :param track_id: audio file name, or identifier of the musical execution
    :param label: music or clique identification
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params). Shall include the sample_rate (if omitted default=44100), hop_length (if omitted default=512), and a dictionary of features to be extracted

    :return: a python dictionary with all the requested features computed as key, value pairs.
    """
    feature = AudioFeatures(audio_vector=audio_vector, read_file=False, hop_length=params['hop_length'], sample_rate=params['sample_rate'])
    out_dict = dict()
    # now we compute all the listed features in the profile dict and store the results to a output dictionary
    #print("compute_features_vector features: %s, sample_rate: %d, hop_length: %d"%(params['features'], params['sample_rate'], params['hop_length']))
    for method in params['features']:
        out_dict[method] = getattr(feature, method)(params=params[method]) #Atila: adicionei para ficar equivalente a do compute_features original.

    #track_id = os.path.basename(audio_path).replace(params['input_audio_format'], '')
    out_dict['track_id'] = track_id

    label = audio_path.split('/')[-2]
    out_dict['label'] = label
    out_dict['params'] = params

    return out_dict
    
    
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
    #Atila: adicionei hop_length na chamada, para garantir uso do valor cofigurado no PROFILE
    if 'hop_length' not in params.keys():
        hop_length = 512
    else:
        hop_length=params['hop_length']
    feature = AudioFeatures(audio_file=audio_path, hop_length=hop_length, sample_rate=params['sample_rate'], verbose=params['verbose'])
    if feature.audio_vector.shape[0] == 0:
        raise IOError("Empty or invalid audio recording file -%s-" % audio_path)

    if params['endtime']:
        feature.audio_vector = feature.audio_slicer(endTime=params['endtime'])
    if params['downsample_audio']:
        feature.audio_vector = feature.resample_audio(params['sample_rate'] / params['downsample_factor'])

    out_dict = dict()
    # now we compute all the listed features in the profile dict and store the results to a output dictionary
    for method in params['features']:
        #print("vamos calcular feature %s"%method)
        out_dict[method] = getattr(feature, method)(params=params[method]) #Dirceu adicionou params=params...

    track_id = os.path.basename(audio_path).replace(params['input_audio_format'], '')
    out_dict['track_id'] = track_id

    label = audio_path.split('/')[-2]
    out_dict['label'] = label

    out_dict['params'] = params #Dirceu adicionou key 'params' ao dicionario de features

    return out_dict


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
    #Atila: modifiquei para verificar se o arquivo de features já existe, e em caso positivo, pula a extração de features para esse arquivo

    start_time = time.perf_counter()
    _LOG_FILE.info("Extracting features for %s " % input_txt_file)
    data = read_txt_file(input_txt_file)
    data = [path for path in data if os.path.exists(path)]
    if len(data) < 1:
        _LOG_FILE.debug("Empty collection txt file -%s- !" % input_txt_file)
        raise IOError("Empty collection txt file -%s- !" % input_txt_file)

    if params['verbose'] >= 1: #Dirceu: so mostra progress bar se params['verbose']>=1
        progress_bar = Bar('acoss.extractor.compute_features_from_list_file',
                            max=len(data),
                            suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    for song in data:
        try:
            work_id = song.split('/')[-2]
            work_dir = "%s%s/" % (feature_dir, work_id)
            feature_file_name = work_dir + os.path.basename(song).replace(params['input_audio_format'], '') + '.h5'
            if params['overwrite'] or not os.path.exists(feature_file_name):
                #print("vamos calcular features")
                feature_dict = compute_features(audio_path=song, params=params)
                #print("vamos ver se existe a pasta")
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir)
                # save as h5
                #print("Vamos salvar %s"%feature_file_name)
                dd.io.save(feature_file_name, feature_dict)
            else:
                print("Feature file %s already exists. Skipping\n"% feature_file_name)
        except:
            _ERRORS.append(input_txt_file)
            _ERRORS.append(song)
            _LOG_FILE.debug("Error: skipping computing features for audio file --%s-- " % song)
            print("Unexpected error:", sys.exc_info()[0])
        if params['verbose'] >= 1:
            progress_bar.next()
    if params['verbose'] >= 1:
        progress_bar.finish()
    _LOG_FILE.info("Process finished in - %s - seconds" % (time.perf_counter() - start_time))
    
@ray.remote  #Dirceu: ray
def compute_features_from_list_file_ray(input_txt_file, feature_dir, params=PROFILE):
    compute_features_from_list_file(input_txt_file, feature_dir = feature_dir, params=params)

def batch_feature_extractor(dataset_csv, audio_dir, feature_dir, n_workers=-1, mode='parallel', params=PROFILE):
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
    start_time = time.perf_counter()
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    batch_file_dir = "./batches/"
    create_audio_path_batches(dataset_csv,
                              dir_to_save=batch_file_dir,
                              root_audio_dir=audio_dir,
                              audio_format=params['input_audio_format'])

    collection_files = glob.glob(batch_file_dir + '*.txt')
    #feature_path = [feature_dir for i in range(len(collection_files))]
    #param_list = [params for i in range(len(collection_files))]
    #args = zip(collection_files, feature_path, param_list)
    _LOG_FILE.info("Computing batch feature extraction using '%s' mode the profile: %s \n" % (mode, params))
    if mode == 'parallel': #Dirceu para usar ray
        # Atila: para usar Ray de modo mais eficiente
        tic = time.perf_counter()

        #progressbar = Bar('acoss.extractor.batch_feature_extractor %d CPUs'%n_workers, 
        #        max=len(collection_files), 
        #        suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        remaining_ids = []
        if n_workers > len(collection_files):  #Verificando se por acaso temos mais cores do que tarefas. 
            n_proc = len(collection_files)  # Nesse caso vamos disparar menos processos
        else:
            n_proc = n_workers
        print("Paralelo com Ray. Vamos usar %d cores."%n_proc)

        ray.init(num_cpus=n_proc)
        params_id = ray.put(params)
        features_dir_id = ray.put(feature_dir)
        for k in range(0,n_proc):       # loop sem wait, para iniciar processos
            remaining_ids.append(compute_features_from_list_file_ray.remote(collection_files[k],features_dir_id,params_id))  # Vamos disparar tantos processos quantos cores tenhamos
        #print("Todos os cores iniciados e executando. Vamos ao wait.")
        for k in range(n_proc, len(collection_files)): # loop com wait, para seguir e tratar demais tarefas
            ready_ids, remaining_ids = ray.wait(remaining_ids)  #Esperando algum core terminar
            #print("Objeto OID: %s, k: %d pronto para a proxima."%(ready_ids,k))
            remaining_ids.append(compute_features_from_list_file_ray.remote(collection_files[k],features_dir_id,params_id))  #Vagou. Vamos enviar proxima tarefa
            #progressbar.next()
        while len(remaining_ids):
            ready_ids, remaining_ids = ray.wait(remaining_ids)
            #progressbar.next()
        #progressbar.finish()
        ray.shutdown()
        # Atila: fim uso Ray
        """
        Codigo do Dirceu
        music_groups = split_list_with_N_elements(collection_files,n_workers)

        for cpaths in music_groups:
            params_id = ray.put(params)
            features_dir_id = ray.put(feature_dir)
            ray.get([compute_features_from_list_file.remote(cpath,features_dir_id,params_id) for cpath in cpaths])
        """
        #Parallel(n_jobs=n_workers, verbose=1)(delayed(compute_features_from_list_file)\
        #                                      (cpath, fpath, param) for cpath, fpath, param in args)
    elif mode == 'single':
        tic = time.perf_counter()
        progressbar = Bar('acoss.extractor.batch_feature_extractor Single CPU', 
                        max=len(collection_files), 
                        suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for cpath in collection_files:
            compute_features_from_list_file(cpath, feature_dir, params)
            progressbar.next()
        """
        for cpath, fpath, param in args:
            compute_features_from_list_file(cpath, fpath, param)
            progressbar.next()
        """
        progressbar.finish()
        _LOG_FILE.info("Single mode feature extraction finished in %s" % (time.monotonic() - tic))
    else:
        raise IOError("Wrong value for the parameter 'mode'. Should be either 'single' or 'parallel'")
    stop_time = time.perf_counter()
    print("Finished batch_feature_extractor in %s" % (stop_time - start_time))
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
    parser.add_argument("-f", "--feature_list", action="store", type=str, 
                        default="[hpcp,key_extractor,librosa_onsets,madmom_features,mfcc_htk]",
                        help="List of features to compute. Eg. [hpcp,crema]")
    parser.add_argument("-m", "--run_mode", action="store", default='parallel',
                        help="Whether to run the extractor in single or parallel mode. "
                             "Choose one of ['single', 'parallel']")
    parser.add_argument("-n", "--workers", action="store",type=int, default=-1,
                        help="No of workers for running the batch extraction process. Only valid in 'parallel' mode.")
    parser.add_argument("-v", "--verbose", action="store", type=int, default=0,
                        help="It shows the progress bar.")
    parser.add_argument("-r", "--overwrite", action="store", type=int, default=1,
                        help="Overwrite existing files (0, 1) - default 1.")


    cmd_args = parser.parse_args()

    print("Args: %s" % cmd_args)

    if not os.path.exists(cmd_args.feature_dir):
        os.mkdir(cmd_args.feature_dir)

    feature_list = cmd_args.feature_list.strip('][').split(',')
    updated_profile = PROFILE.copy()
    del updated_profile['features']
    updated_profile['features'] = feature_list
    updated_profile['verbose'] = cmd_args.verbose
    updated_profile['overwrite'] = cmd_args.overwrite
    num_cpus = cmd_args.workers
    # Prepara para Ray
    if cmd_args.run_mode=='parallel':
        if cmd_args.workers == -1:
            num_cpus = psutil.cpu_count(logical=False)
        else:
            num_cpus = cmd_args.workers
        print("Extração em paralelo, usando: %d CPUs" % num_cpus)


    batch_feature_extractor(dataset_csv=cmd_args.dataset_csv,
                            audio_dir=cmd_args.audio_dir,
                            feature_dir=cmd_args.feature_dir,
                            n_workers=num_cpus,
                            mode=cmd_args.run_mode,
                            params=updated_profile)

    print("... Done ....")
    print(" -- PROFILE INFO -- \n %s" % updated_profile)
