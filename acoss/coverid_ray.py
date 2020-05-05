# -*- coding: utf-8 -*-
"""
Interface to run the various cover id algorithms for acoss benchmarking.
"""
import argparse
import time
import sys
import os
from shutil import rmtree
import ray
from progress.bar import Bar
import psutil
import glob
import numpy as np

from acoss.utils import (log, read_txt_file, create_csv_cliques_and_batches, savelist_to_file,
                         split_list_with_N_elements, create_dataset_filepaths)
from acoss.algorithms.algorithm_template import CoverAlgorithm

__all__ = ['benchmark', 'algorithm_names']

_LOG_FILE_PATH = "acoss.coverid.log"
_LOGGER = log(_LOG_FILE_PATH)
_ERRORS = list()

# list the available cover song identification algorithms in acoss
algorithm_names = ["Serra09", "EarlyFusionTraile", "LateFusionChen", "FTM2D", "SiMPle"]



@ray.remote
def compute_groups_from_list_file(input_txt_file,
                                  feature_dir,
                                  results_dir='../result',
                                  algorithm='Serra09',
                                  shortname='covers80',
                                  chroma_type='hpcp',
                                  verbose=0):
    """
    Compute specified audio features for a list of audio file paths and store to disk as .h5 file
    from a given input text file.
    It is a wrapper around 'compute_features'.

    :param input_txt_file: a text file with a list of audio file paths
    :param feature_dir: a path
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params)

    :return: None
    """
    progress_bar=None
    start_time = time.monotonic()
    _LOGGER.info("Extracting Batch File: %s " % input_txt_file)
    data = read_txt_file(input_txt_file)
    data = [path for path in data if os.path.exists(path)]
    if len(data) < 1:
        _LOGGER.debug("Empty collection txt file -%s- !" % input_txt_file)
        raise IOError("Empty collection txt file -%s- !" % input_txt_file)

    if verbose >= 1:
        progress_bar = Bar('acoss.coverid_ray.compute_groups_from_list_file',
                            max=len(data),
                            suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    for csv in data:
        shortname = os.path.splitext(os.path.basename(csv))[0]
        results_path = os.path.join(results_dir,shortname)
        _LOGGER.info("computing Similarity for %s " % csv)
        #try:
        benchmark(csv,
                  feature_dir,
                  cache_dir=results_path,
                  chroma_type=chroma_type,
                  algorithm=algorithm,
                  shortname=shortname,
                  parallel=False,
                  n_workers=1)

            #save as h5
            #dd.io.save(filename,feature_dict)
        # except:
        #     _ERRORS.append(input_txt_file)
        #     _ERRORS.append(csv)
        #     _LOGGER.debug("Error: skipping computing features for audio file --%s-- " % csv)

        if verbose >= 1:
            progress_bar.next()

    if verbose >= 1:
        progress_bar.finish()

    _LOGGER.info("Process finished in - %s - seconds" % (start_time - time.time()))


def batch_groups(dataset_csv,
                 feature_dir,
                 cache_dir,
                 save_csv_dir,
                 batchesdir,
                 shortname='covers80',
                 mode='parallel',
                 algorithm="Serra09",
                 chroma_type='hpcp',
                 n_workers=-1):
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
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    batch_file_dir = batchesdir

    create_csv_cliques_and_batches(dataset_csv, save_csv_dir, batchesdir, n_workers=n_workers)

    collection_files = glob.glob(batch_file_dir + '*.txt')
    _LOGGER.info("Computing batch similarity using '%s' mode \n" % (mode))
    print(collection_files)

    if mode == 1:
        music_groups = split_list_with_N_elements(collection_files,n_workers)
        print(music_groups)

        for cpaths in music_groups:
            _LOGGER.info("Computing similarity using Ray '%s' \n" % (cpaths))
            algorithm_id = ray.put(algorithm)
            features_dir_id = ray.put(feature_dir)
            chroma_type_id = ray.put(chroma_type)
            cache_dir_id = ray.put(cache_dir)

            ray.get([compute_groups_from_list_file.remote(cpath,features_dir_id, cache_dir_id,algorithm_id,chroma_type_id) for cpath in cpaths])

    elif mode == 0:
        tic = time.monotonic()
        for cpath in collection_files:
            algorithm_id = ray.put(algorithm)
            features_dir_id = ray.put(feature_dir)
            shortname_id = ray.put(shortname)
            chroma_type_id = ray.put(chroma_type)
            cache_dir_id = ray.put(cache_dir)

            ray.get(compute_groups_from_list_file.remote(cpath,features_dir_id, cache_dir_id,algorithm_id,shortname_id,chroma_type_id))

        _LOGGER.info("Single mode similarity finished in %s" % (time.monotonic() - tic))
    else:
        raise IOError("Wrong value for the parameter 'mode'. Should be either 'single' or 'parallel'")
    savelist_to_file(_ERRORS,'_erros_acoss.coversid.txt')
    #rmtree(batch_file_dir)
    _LOGGER.info("Log file located at '%s'" % _LOG_FILE_PATH)


def benchmark_ray(dataset_csv,
              feature_dir,
              cache_dir='cache',
              chroma_type="hpcp",
              algorithm="Serra09",
              shortname="covers80",
              parallel=True,
              n_workers=-1):
    """Benchmark a specific cover id algorithm with a given input dataset annotation csv file.

    Arguments:
        dataset_csv {string} -- path to dataset csv annotation file
        feature_dir {string} -- path to the directory where the pre-computed audio features are stored

    Keyword Arguments:
        feature_type {str} -- type of audio feature you want to use for benchmarking. (default: {"hpcp"})
        algorithm {str} -- name of the algorithm you want to benchmark (default: {"Serra09"})
        shortname {str} -- description (default: {"DaTacos-Benchmark"})
        parallel {bool} -- whether you want to run the benchmark process with parallel workers (default: {True})
        n_workers {int} -- number of workers required. By default it uses as much workers available on the system. (default: {-1})

    Raises:
        NotImplementedError: when an given algorithm method in not implemented in acoss.benchmark
    """

    if algorithm not in algorithm_names:
        warn = ("acoss.coverid: Couldn't find '%s' algorithm in acoss \
                                Available cover id algorithms are %s "
                % (algorithm, str(algorithm_names)))
        _LOGGER.debug(warn)
        raise NotImplementedError(warn)

    _LOGGER.info("Running acoss cover identification benchmarking for the algorithm - '%s'" % algorithm)

    start_time = time.monotonic()

    if algorithm == "Serra09":
        from .algorithms.rqa_serra09 import Serra09
        # here run the algo
        serra09 = Serra09(dataset_csv=dataset_csv,
                          datapath=feature_dir,
                          chroma_type=chroma_type,
                          shortname=shortname)
        _LOGGER.info('Computing pairwise similarity...')
        serra09.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        serra09.normalize_by_length()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in serra09.Ds.keys():
            serra09.getEvalStatistics(similarity_type)
        serra09.cleanup_memmap()

    elif algorithm == "EarlyFusionTraile":
        from acoss.algorithms.earlyfusion_traile import EarlyFusion

        _LOGGER.info('Starting EarlyFusionTraile...')

        # get basic variables
        CA = CoverAlgorithm(dataset_csv,
                            name=algorithm,
                            datapath=feature_dir,
                            shortname=shortname,
                            cachedir=cache_dir,
                            similarity_types=["mfccs", "ssms", "chromas", "early"])

        filepaths_id = ray.put(CA.filepaths)
        workers = []
        for k in range(n_workers):
            EF_ID = EarlyFusion.remote(dataset_csv=dataset_csv,
                               datapath=feature_dir,
                               cache_dir=cache_dir,
                               chroma_type=chroma_type,
                               shortname=shortname,
                               K=10,
                               filepaths=filepaths_id,
                               create_Ds=False)
            workers.append(EF_ID)

        _LOGGER.info('Feature loading done...')

        n_chunks = n_workers
        chunks = np.array_split(range(len(CA.filepaths)), n_chunks)
        works_ids = []
        w = 0
        for chunk in chunks:
            works_ids.append(workers[w].load_features_in_block.remote(chunk, keep_features_in_memory=False))
            w = (w + 1) % n_workers

        while len(works_ids) > 0:
            done_work, works_ids = ray.wait(works_ids)

        _LOGGER.info('Computing pairwise similarity...')

        chunks = ray.get(workers[0].generate_pairs.remote(n_chunks=n_chunks,symmetric=True))

        w = 0
        works_simm = []
        for chunk in chunks:
            works_simm.append(workers[w].similarity.remote(chunk))
            w = (w+1)%n_workers
        count = 0
        while len(works_simm):
            done_work, works_simm = ray.wait(works_simm)
            results = ray.get(done_work)[0]
            CA.set_Ds_results(results)

        DS_ID = ray.put(CA.get_Ds())

        workers[0].set_Ds.remote(DS_ID)
        ray.get(workers[0].get_all_clique_ids.remote())
        ray.get(workers[0].apply_simmetry.remote(symmetric=True))

        # early_fusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        ray.get(workers[0].do_late_fusion.remote())

        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)

        work_eval = [workers[0].getEvalStatistics.remote(similarity_type) for similarity_type in CA.get_Ds().keys()]
        ray.get(work_eval)

        CA.cleanup_memmap()

    elif algorithm == "LateFusionChen":
        from .algorithms.latefusion_chen import ChenFusion

        chenFusion = ChenFusion(dataset_csv=dataset_csv,
                                datapath=feature_dir,
                                chroma_type=chroma_type,
                                shortname=shortname)
        _LOGGER.info('Computing pairwise similarity...')
        chenFusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True, verbose=False)
        chenFusion.normalize_by_length()
        chenFusion.do_late_fusion()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in chenFusion.Ds.keys():
            _LOGGER.info(similarity_type)
            chenFusion.getEvalStatistics(similarity_type)
        chenFusion.cleanup_memmap()

    elif algorithm == "FTM2D":
        from .algorithms.ftm2d import FTM2D

        ftm2d = FTM2D(dataset_csv=dataset_csv,
                      datapath=feature_dir,
                      chroma_type=chroma_type,
                      shortname=shortname)
        for i in range(len(ftm2d.filepaths)):
            ftm2d.load_features(i)
        _LOGGER.info('Feature loading done...')
        _LOGGER.info('Computing pairwise similarity...')
        ftm2d.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in ftm2d.Ds.keys():
            ftm2d.getEvalStatistics(similarity_type)
        ftm2d.cleanup_memmap()

    elif algorithm == "SiMPle":
        from .algorithms.simple_silva import Simple

        simple = Simple(dataset_csv=dataset_csv,
                        datapath=feature_dir,
                        chroma_type=chroma_type,
                        shortname=shortname)
        for i in range(len(simple.filepaths)):
            simple.load_features(i)
        _LOGGER.info('Feature loading done...')
        _LOGGER.info('Computing pairwise similarity...')
        simple.all_pairwise(parallel, n_cores=n_workers, symmetric=False)
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in simple.Ds.keys():
            simple.getEvalStatistics(similarity_type)
        simple.cleanup_memmap()

    _LOGGER.info("acoss.coverid benchmarking finsihed in %s" % (time.monotonic() - start_time))
    _LOGGER.info("Log file located at '%s'" % _LOG_FILE_PATH)


def benchmark(dataset_csv,
              feature_dir,
              cache_dir='cache',
              chroma_type="hpcp",
              algorithm="Serra09",
              shortname="covers80",
              parallel=True,
              n_workers=-1):
    """Benchmark a specific cover id algorithm with a given input dataset annotation csv file.
    
    Arguments:
        dataset_csv {string} -- path to dataset csv annotation file
        feature_dir {string} -- path to the directory where the pre-computed audio features are stored
    
    Keyword Arguments:
        feature_type {str} -- type of audio feature you want to use for benchmarking. (default: {"hpcp"})
        algorithm {str} -- name of the algorithm you want to benchmark (default: {"Serra09"})
        shortname {str} -- description (default: {"DaTacos-Benchmark"})
        parallel {bool} -- whether you want to run the benchmark process with parallel workers (default: {True})
        n_workers {int} -- number of workers required. By default it uses as much workers available on the system. (default: {-1})
    
    Raises:
        NotImplementedError: when an given algorithm method in not implemented in acoss.benchmark
    """

    if algorithm not in algorithm_names:
        warn = ("acoss.coverid: Couldn't find '%s' algorithm in acoss \
                                Available cover id algorithms are %s "
                                % (algorithm, str(algorithm_names)))
        _LOGGER.debug(warn)
        raise NotImplementedError(warn)

    _LOGGER.info("Running acoss cover identification benchmarking for the algorithm - '%s'" % algorithm)

    start_time = time.monotonic()

    if algorithm == "Serra09":
        from .algorithms.rqa_serra09 import Serra09
        # here run the algo
        serra09 = Serra09(dataset_csv=dataset_csv,
                          datapath=feature_dir,
                          chroma_type=chroma_type,
                          shortname=shortname)
        _LOGGER.info('Computing pairwise similarity...')
        serra09.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        serra09.normalize_by_length()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in serra09.Ds.keys():
            serra09.getEvalStatistics(similarity_type)
        serra09.cleanup_memmap()

    elif algorithm == "EarlyFusionTraile":
        from acoss.algorithms.earlyfusion_traile import EarlyFusion
        _LOGGER.info('Starting EarlyFusionTraile...')
        early_fusion = EarlyFusion.remote(dataset_csv=dataset_csv,
                                   datapath=feature_dir,
                                   cache_dir=cache_dir,
                                   chroma_type=chroma_type,
                                   shortname=shortname,
                                   K=10)
        _LOGGER.info('Feature loading done...')

        filepaths = ray.get(early_fusion.get_filepaths.remote())
        print("filepaths: ",len(filepaths))
        works_ids = [early_fusion.load_features.remote(i,keep_features_in_memory=False) for i in range(len(filepaths))]
        print("works_ids",len(works_ids))

        while len(works_ids)>0:
            done_work, rest = ray.wait(works_ids)
            works_ids=rest

        _LOGGER.info('Computing pairwise similarity...')

        chunks = ray.get(early_fusion.generate_pairs.remote(symmetric=True))
        works = [early_fusion.compute_similarity.remote(chunk) for chunk in chunks]
        while len(works):
            done_work, works = ray.wait(works)
        ray.get(early_fusion.get_all_clique_ids.remote())
        ray.get(early_fusion.apply_simmetry.remote(symmetric=True))

        #early_fusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        ray.get(early_fusion.do_late_fusion.remote())

        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        Ds_Types = ray.get(early_fusion.get_Ds_types.remote())
        work = [early_fusion.getEvalStatistics.remote(similarity_type) for similarity_type in Ds_Types]
        ray.get(work)
        ray.get(early_fusion.cleanup_memmap.remote())

    elif algorithm  == "LateFusionChen":
        from .algorithms.latefusion_chen import ChenFusion

        chenFusion = ChenFusion(dataset_csv=dataset_csv,
                                datapath=feature_dir,
                                chroma_type=chroma_type,
                                shortname=shortname)
        _LOGGER.info('Computing pairwise similarity...')
        chenFusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True, verbose=False)
        chenFusion.normalize_by_length()
        chenFusion.do_late_fusion()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in chenFusion.Ds.keys():
            _LOGGER.info(similarity_type)
            chenFusion.getEvalStatistics(similarity_type)
        chenFusion.cleanup_memmap()

    elif algorithm == "FTM2D":
        from .algorithms.ftm2d import FTM2D

        ftm2d = FTM2D(dataset_csv=dataset_csv,
                      datapath=feature_dir,
                      chroma_type=chroma_type,
                      shortname=shortname)
        for i in range(len(ftm2d.filepaths)):
            ftm2d.load_features(i)
        _LOGGER.info('Feature loading done...')
        _LOGGER.info('Computing pairwise similarity...')
        ftm2d.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in ftm2d.Ds.keys():
            ftm2d.getEvalStatistics(similarity_type)
        ftm2d.cleanup_memmap()

    elif algorithm == "SiMPle":
        from .algorithms.simple_silva import Simple

        simple = Simple(dataset_csv=dataset_csv,
                        datapath=feature_dir,
                        chroma_type=chroma_type,
                        shortname=shortname)
        for i in range(len(simple.filepaths)):
            simple.load_features(i)
        _LOGGER.info('Feature loading done...')
        _LOGGER.info('Computing pairwise similarity...')
        simple.all_pairwise(parallel, n_cores=n_workers, symmetric=False)
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in simple.Ds.keys():
            simple.getEvalStatistics(similarity_type)
        simple.cleanup_memmap()

    _LOGGER.info("acoss.coverid benchmarking finsihed in %s" % (time.monotonic() - start_time))
    _LOGGER.info("Log file located at '%s'" % _LOG_FILE_PATH)


def parser_args(cmd_args):

    parser = argparse.ArgumentParser(sys.argv[0], description="Benchmark a specific cover id algorithm with a given"
                                                              "input dataset csv annotations",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '--dataset_csv', type=str, action="store",
                        help="Input dataset csv file")
    parser.add_argument("-d", '--feature_dir', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-r", '--cache_dir', type=str, action="store", default='../cache',
                        help="Path to cache files")
    parser.add_argument("-v", '--csv_dir', type=str, action="store", default='../csv',
                        help="Path to csv temporary files")
    parser.add_argument("-b", '--batches_dir', type=str, action="store", default='../batches',
                        help="Path to batch temporary files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="covers80",
                        help="Short name for dataset")
    parser.add_argument("-a", "--algorithm", type=str, action="store", default="Serra09",
                        help="Algorithm to use")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default="hpcp",
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=1,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_workers', type=int, action="store", default=-1,
                        help="No of workers required for parallelization")
    parser.add_argument("-t", "--type_cluster", action="store", type=int, default=0,
                        help="Cluster type to use (0 - without cluster, 1 - Slurm) - default 0.")
    parser.add_argument("-w", "--redis_password", action="store", type=str, default="",
                        help="Cluster redisPassword - generated by slurm script.")

    return parser.parse_args(cmd_args)


if __name__ == '__main__':

    args = parser_args(sys.argv[1:])

    # args.dataset_csv = 'data/Covers10k_p2.csv'
    # args.feature_dir='/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/features/'
    # args.results_dir='/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/results/'
    # args.csv_dir='/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/csv/'
    # args.batches_dir='/mnt/Data/dirceusilva/Results/features/Covers10k_11khz/batches/'
    # args.algorithm='EarlyFusionTraile'
    # args.chroma_type='hpcp'
    # args.parallel=0
    # args.n_workers=8
    # args.type_cluster=0
    # args.shortname="covers10k"

    # Start Ray
    useRay = True

    num_cpus = args.n_workers
    if args.n_workers == -1:
        num_cpus = psutil.cpu_count(logical=False)

    print("Nr CPUs: %d" % num_cpus)
    if useRay:
        if args.type_cluster == 1:  # Slurm Cluster Configuration
            ray.init(address=os.environ["ip_head"],
                     redis_password=args.redis_password)  # 1 GB
        else:
            ray.init(num_cpus=num_cpus, memory=int(2*1024*1024*1024), object_store_memory=(300*1024*1024))

    # if useRay or args.type_cluster == 1:
    #
    #     batch_groups(dataset_csv=args.dataset_csv,
    #                  feature_dir=args.feature_dir,
    #                  cache_dir=args.cache_dir,
    #                  save_csv_dir=args.csv_dir,
    #                  batchesdir=args.batches_dir,
    #                  shortname=args.shortname,
    #                  mode=args.parallel,
    #                  algorithm=args.algorithm,
    #                  chroma_type=args.chroma_type,
    #                  n_workers=num_cpus)
    # else:
    benchmark_ray(dataset_csv=args.dataset_csv,
                  cache_dir=args.cache_dir,
                  feature_dir=args.feature_dir,
                  chroma_type=args.chroma_type,
                  algorithm=args.algorithm,
                  shortname=args.shortname,
                  parallel=bool(args.parallel),
                  n_workers=args.n_workers)
