# -*- coding: utf-8 -*-
"""
Some general utility functions used in acoss
"""
import logging
import time
import json
import os
import numpy as np
import pandas as pd
from shutil import rmtree

from acoss import __path__ as acoss_pkg_path

ACOSS_PATH = acoss_pkg_path[0]
# csv files with annotations required for acoss benchmarking for da-tacos benchmark subset 
# https://mtg.github.io/da-tacos/
DA_TACOS_BENCHMARK_CSV = os.path.join(ACOSS_PATH,"data/da-tacos_benchmark_subset.csv")
# csv files with annotations required for acoss benchmarking for covers80 dataset 
# https://labrosa.ee.columbia.edu/projects/coversongs/covers80/
COVERS_80_CSV = os.path.join(ACOSS_PATH, "data/covers80_annotations.csv")


def log(log_file):
    """Returns a logger object with predefined settings"""
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger


def timeit(method):
    """A custom timeit profiling decorator"""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r - runtime : %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def read_txt_file(txt_file):
    """read a text file and strips \n char from it"""
    f = open(txt_file)
    data = f.readlines()
    return [d.replace('\n', '') for d in data]


def savelist_to_file(path_list, filename):
    doc = open(filename, 'w')
    for item in path_list:
        doc.write("%s\n" % item)
    doc.close()


def create_audio_path_batches(dataset_csv, dir_to_save, root_audio_dir="./", audio_format=".mp3", reset=False, n_workers=None):
    """Create batches of audio file paths for batch processing given a input dataset_csv annotation"""
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)
    elif reset:
        rmtree(dir_to_save)

    data_paths = create_dataset_filepaths(dataset_csv, root_audio_dir, audio_format)
    if n_workers:
        batch_size = min(n_workers,len(data_paths))
    else:
        batch_size = len(data_paths)

    for path in data_paths:
        if not os.path.exists(path):
            raise Exception(".. Invalid audio filepath -- %s -- found in the collection file" % path)
    if batch_size > len(data_paths):
        raise UserWarning("Batch size shouldn't be greater than the size of audio file")
    song_chunks = np.array_split(data_paths, batch_size)
    for idx, chunk in enumerate(song_chunks):
        savelist_to_file(chunk, dir_to_save + str(idx) + '_batch.txt')


def create_csv_cliques_and_batches(dataset_csv, dir_save_csv, batches_dir, n_workers=None):
    """Create csv files to each Covers groups """

    if not os.path.exists(dir_save_csv):
        os.mkdir(dir_save_csv)
    elif reset:
        rmtree(dir_save_csv)

    allcliques = pd.read_csv(dataset_csv, encoding="ISO-8859-1", dtype=str)
    aftergrouping_cliques = allcliques.groupby("work_id")
    data_paths = []

    for name, group in aftergrouping_cliques:
        filesave = os.path.join(dir_save_csv, name + '.csv')
        group.to_csv(filesave, index=False)
        data_paths.append(filesave)

    if not os.path.exists(batches_dir):
        os.mkdir(batches_dir)
    elif reset:
        rmtree(batches_dir)

    if n_workers:
        batch_size = min(n_workers,len(data_paths))
    else:
        batch_size = len(data_paths)

    for path in data_paths:
        if not os.path.exists(path):
            raise Exception(".. Invalid audio filepath -- %s -- found in the collection file" % path)
    if batch_size > len(data_paths):
        raise UserWarning("Batch size shouldn't be greater than the size of audio file")
    song_chunks = np.array_split(data_paths, batch_size)
    for idx, chunk in enumerate(song_chunks):
        savelist_to_file(chunk, batches_dir + str(idx) + '_batch.txt')


def create_dataset_filepaths(dataset_csv, root_audio_dir, file_format=".mp3"):
    """Constructs audio file paths from dataset csv annotation.
    eg: For a csv file with two columns `work_id, track_id`
        the audio path will be "root_audio_dir/work_id/track_id.mp3"
    """
    dataset = pd.read_csv(dataset_csv, dtype=str)

    warn = ("Wrong input dataset csv annotation file '%s'. Expected a csv file with the columns of key 'work_id', 'track_id'" 
            % dataset_csv)

    for key in dataset.keys().tolist():
        if key not in ['work_id', 'track_id']:
            raise IOError(warn)

    dataset['filepath'] = dataset.apply(lambda x: root_audio_dir + str(x.work_id) + "/" + str(x.track_id) + file_format, axis=1)
    return dataset.filepath.tolist()


def da_tacos_metadata_to_acoss_csv(datacos_metadata_json, output_csv):
    """Parse da-tacos dataset metadata json file to csv file required for acoss"""
    with open(datacos_metadata_json) as f:
        dataset = json.load(f)
    work_ids = list()
    perf_ids = list()
    for work_id in dataset.keys():
        for perf_id in dataset[work_id].keys():
            perf_ids.append(perf_id)
            work_ids.append(work_id)
    df = pd.DataFrame({'work_id': work_ids, 'track_id': perf_ids})
    df.to_csv(output_csv, index=False)


def generate_covers80_acoss_csv(covers80_audio_data_path, output_csv):
    """Generate acoss csv annotation file from covers80 audio data folder"""
    work_ids = list()
    track_ids = list()
    for work_label in os.listdir(covers80_audio_data_path):
        work_path = os.path.join(covers80_audio_data_path, work_label)
        if os.path.isdir(work_path):
            for track in os.listdir(work_path):
                if track.endswith('.mp3'):
                    track_ids.append(track.replace('.mp3', ''))
                    work_ids.append(work_label)
    
    df = pd.DataFrame({'work_id': work_ids, 'track_id': track_ids})
    df.to_csv(output_csv, index=False)


def split_list_with_N_elements(seq,n):
    #spli a list in sublists with n elements + the reminder

    newlist = [seq[i * n:(i + 1) * n] for i in range((len(seq) + n - 1) // n )]
    return newlist

def get_index_split_list_with_N_elements(seq,n):
    #spli a list in sublists with n elements + the reminder

    newlist = [seq[i * n:(i + 1) * n] for i in range((len(seq) + n - 1) // n )]
    return newlist



def split_list_in_CoversGroups(filepaths):
    """
    Get the row and column index ranges of blocks in an
    all pairs of the same clique similarity experiment between N songs
    :param filepaths: The list of songs, with their path name as an identifier of a clique
    :returns ranges: An array of ranges [[starti, endi, startj, endj]]
    	comprising each block.
    """
    ranges = []
    n_cn = ""
    i1 = i2 = 0
    for fname in allFiles:
        cn = fname.split("/")[-2]
        if cn != n_cn:
            if n_cn != "":
                ranges.append([i1, i2])
            i1 = i2
            n_cn = cn
        i2 += 1
    ranges.append([i1, i2])
    return ranges


def split_list_in_N_parts(seq, n):
    # split a list in N parts + the reminder
    avg = len(seq) / float(n)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out