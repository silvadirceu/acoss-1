# -*- coding: utf-8 -*-
"""
Interface to run the various cover id algorithms for acoss benchmarking.
"""
# Standard library imports
import argparse
import os
from shutil import rmtree
import sys
import time

# Third party imports

# Local application imports
from acoss.utils import log

__all__ = ['benchmark', 'algorithm_names']

_LOG_FILE_PATH = "acoss.coverid.log"
_LOGGER = log(_LOG_FILE_PATH)

# list the available cover song identification algorithms in acoss
algorithm_names = ["Serra09", "EarlyFusionTraile", "LateFusionChen", "FTM2D", "SiMPle"]

def benchmark(dataset_csv,
              feature_dir,
              cache_dir="./cache",
              feature_type="hpcp",
              onset_source = "madmom_features",
              algorithm="Serra09",
              shortname="covers80",
              parallel=True,
              n_workers=-1):  #Atila: acrescentei parametro onset_lib
    """Benchmark a specific cover id algorithm with a given input dataset annotation csv file.
    
    Arguments:
        dataset_csv {string} -- path to dataset csv annotation file
        feature_dir {string} -- path to the directory where the pre-computed audio features are stored
    
    Keyword Arguments:
        feature_type {str} -- type of audio feature you want to use for benchmarking. (default: {"hpcp"})
        onset_source(str_ - onset library to be used (default: {'madmom_features'})
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
        from acoss.algorithms.rqa_serra09 import Serra09
        # here run the algo
        serra09 = Serra09(dataset_csv=dataset_csv,
                          datapath=feature_dir,
                          chroma_type=feature_type,
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

        early_fusion = EarlyFusion(dataset_csv=dataset_csv,
                                   datapath=feature_dir,
                                   chroma_type=feature_type,
                                   onset_lib=onset_source,
                                   shortname=shortname)
        _LOGGER.info('Class loading done...')
        for i in range(len(early_fusion.filepaths)):
            early_fusion.load_features(i)
        _LOGGER.info('Feature loading done...')
        _LOGGER.info('Computing pairwise similarity...')
        early_fusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        early_fusion.do_late_fusion()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in early_fusion.Ds:
            early_fusion.getEvalStatistics(similarity_type)
        early_fusion.cleanup_memmap()

    elif algorithm  == "LateFusionChen":
        from acoss.algorithms.latefusion_chen import ChenFusion

        chenFusion = ChenFusion(dataset_csv=dataset_csv,
                                datapath=feature_dir,
                                chroma_type=feature_type,
                                shortname=shortname)
        _LOGGER.info('Computing pairwise similarity...')
        chenFusion.all_pairwise(parallel, n_cores=n_workers, symmetric=True)
        chenFusion.normalize_by_length()
        chenFusion.do_late_fusion()
        _LOGGER.info('Running benchmark evaluations on the given dataset - %s' % dataset_csv)
        for similarity_type in chenFusion.Ds.keys():
            _LOGGER.info(similarity_type)
            chenFusion.getEvalStatistics(similarity_type)
        chenFusion.cleanup_memmap()

    elif algorithm == "FTM2D":
        from acoss.algorithms.ftm2d import FTM2D

        ftm2d = FTM2D(dataset_csv=dataset_csv,
                      datapath=feature_dir,
                      chroma_type=feature_type,
                      onset_lib=onset_source,
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
        from acoss.algorithms.simple_silva import Simple

        simple = Simple(dataset_csv=dataset_csv,
                        datapath=feature_dir,
                        chroma_type=feature_type,
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Benchmark a specific cover id algorithm with a given"
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
    parser.add_argument("-o", '--onset_source', type=str, action="store", default="madmom_features",
                        help="Onsets library source to be used (deafult=madmom_features. Alternative is librosa_onsets)")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=1,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_workers', type=int, action="store", default=-1,
                        help="No of workers required for parallelization")
    parser.add_argument("-t", "--type_cluster", action="store", type=int, default=0,
                        help="Cluster type to use (0 - without cluster, 1 - Slurm) - default 0.")
    parser.add_argument("-w", "--redis_password", action="store", type=str, default="",
                        help="Cluster redisPassword - generated by slurm script.")

    args = parser.parse_args()

    # Start Ray

    num_cpus = args.n_workers
    if args.n_workers == -1:
        num_cpus = psutil.cpu_count(logical=False)

    print("Nr CPUs: %d" % num_cpus)
    if bool(args.parallel):
        if args.type_cluster == 1:  # Slurm Cluster Configuration
            ray.init(address=os.environ["ip_head"],
                     redis_password=args.redis_password)  # 1 GB
        else:
            ray.init(num_cpus=num_cpus, memory=int(2 * 1024 * 1024 * 1024), object_store_memory=(300 * 1024 * 1024))

    benchmark(dataset_csv=args.dataset_csv,
              feature_dir=args.feature_dir,
              cache_dir=args.cache_dir,
              feature_type=args.chroma_type,
              onset_source=args.onset_source,
              algorithm=args.algorithm,
              shortname=args.shortname,
              parallel=bool(args.parallel),
              n_workers=args.n_workers)
