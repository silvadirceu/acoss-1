# -*- coding: utf-8 -*-
"""
A template class for all benchmark algorithms in `acoss.benchmark` module
"""
# Standard library imports
import glob
import os
import warnings

# Third party imports
import deepdish as dd
import numpy as np
from progress.bar import Bar
import psutil
import ray


# Local application imports
from acoss.utils import create_dataset_filepaths

__all__ = ['CoverAlgorithm']


class CoverAlgorithm(object):
    """
    Attributes
    ----------
    filepaths: list(string)
        List of paths to all files in the dataset
    cliques: {string: set}
        A dictionary of all cover cliques, where the cliques
        index into filepaths
    Ds: {string silarity type: ndarray(num files, num files)}
        A dictionary of pairwise similarity matrices, whose 
        indices index into filepaths.
    """
    def __init__(self,
                 dataset_csv = None,
                 audios_features_vector = None,
                 name="Serra09",
                 datapath="features_benchmark",
                 shortname="full",
                 cachedir="cache",
                 similarity_types=["main"]):
        """
        Parameters
        ----------
        dataset_csv: string
            path to the dataset annotation csv file
        Atila:
        audio_features_vector: list
            list with features of all musics. To compare first music to all others (one to N)
        name: string
            Name of the algorithm
        datapath: string
            Path to folder with h5 files for the benchmark dataset
        shortname: string
            Short name for the dataset (for printing and saving results)
        cachedir: string
            Directory to which to cache intermediate feature computations, etc
        """
        self.name = name
        self.shortname = shortname
        self.cachedir = cachedir
        if dataset_csv is not None:
            self.filepaths = create_dataset_filepaths(dataset_csv, root_audio_dir=datapath, file_format=".h5")
            self.N = len(self.filepaths)
            self.features_vector = None
        elif audios_features_vector is not None:
            self.features_vector = audios_features_vector
            self.filepaths = None
            self.N = len(self.features_vector)  
        self.cliques = {}
        
        if not os.path.exists(cachedir):
            os.mkdir(cachedir)
        self.Ds = {}
        for s in similarity_types:
            self.Ds[s] = np.memmap('%s_%s_dmat' % (self.get_cacheprefix(), s), shape=(self.N, self.N), mode='w+', dtype='float32')
        print("Initialized %s algorithm on %i songs in dataset %s" % (name, self.N, shortname))
    
    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s" % (self.cachedir, self.name, self.shortname)

    def load_features(self, i):
        """
        Load the fields from the h5 file for a particular
        song, and also keep track of which cover clique
        it's in by saving into self.cliques as a side effect
        NOTE: This function can be used to cache information
        about a particular song if that makes comparisons
        faster downstream (e.g. for FTM2D, cache the Fourier
        magnitude shingle median).  But this will not help
        in a parallel scenario
        Parameters
        ----------
        i: int
            Index of song in self.filepaths
        Returns
        -------
        feats: dictionary
            Dictionary of features for the song
        """
        #Atila: modificado para usar dados do vetor de features
        if self.features_vector is not None:
            feats = self.features_vector[i]
        else:
            feats = dd.io.load(self.filepaths[i])
        # Keep track of what cover clique it's in
        if not feats['label'] in self.cliques:
            self.cliques[feats['label']] = set([])
        self.cliques[feats['label']].add(i)
        return feats
    
    def get_all_clique_ids(self, verbose=False):
        """
        Load all h5 files to get clique information as a side effect
        """
        #import os
        if self.filepaths is not None:
            filepath = "%s_clique_info.txt" % self.get_cacheprefix()
            if not os.path.exists(filepath):
                fout = open(filepath, "w")
                for i in range(len(self.filepaths)):
                    feats = CoverAlgorithm.load_features(self, i)
                    if verbose:
                        print(i)
                    print(feats['label'])
                    fout.write("%i,%s\n"%(i, feats['label']))
                fout.close()
            else:
                fin = open(filepath)
                for line in fin.readlines():
                    i, label = line.split(",")
                    label = label.strip()
                    if not label in self.cliques:
                        self.cliques[label] = set([])
                    self.cliques[label].add(int(i))
        else:
            print("Sem filepaths")
            return

    def similarity(self, idxs):
        """
        Given the indices of two songs, return a number
        which is high if the songs are similar, and low
        otherwise, for each similarity type
        Also store this number in D[i, j]
        as a side effect
        Parameters
        ----------
        i: int
            Index of first song in self.filepaths
        j: int
            Index of second song in self.filepaths
        """
        (a, b) = idxs.shape
        for k in range(a):
            i = idxs[k][0]
            j = idxs[k][1]
            score = 0.0
            self.Ds["main"][i, j] = score
    
    @ray.remote
    def similarity_ray(self, idxs):
        """
        Given the indices of two songs, return a number
        which is high if the songs are similar, and low
        otherwise, for each similarity type
        Also store this number in D[i, j]
        as a side effect
        Parameters
        ----------
        i: int
            Index of first song in self.filepaths
        j: int
            Index of second song in self.filepaths
        """
        score={}
        i = idxs[0]
        j = idxs[1]
        score["main"] = 0.0
        #self.Ds["main"][i, j] = score
        return score, i, j
        """
        (a, b) = idxs.shape
        for k in range(a):
            i = idxs[k][0]
            j = idxs[k][1]
            score = 0.0
            #self.Ds["main"][i, j] = score
            return score, i, j
        """


    
    
    def one_N_pairwise(self, parallel=0, n_cores=0):
        """
        Atila:
        Faz comparação das features do primeiro audio num array de features, contra as features de todos os demais no array.
        Opera sobre self.features_vector, chamando self.similarity, que é a rotina que calcula o score entre pares de musicas, de acordo com cada algoritmo.
        Parameters 
        ----------
        parallel: int
            If 0, run serial.  If 1, run parallel
        n_cores: int
            Number of cores to use in a parallel scenario. If -1, all available cores
        """
        from itertools import product
        #all_pairs = [(i, j) for idx, (i, j) in enumerate(combinations(range(len(self.features_vector)), 2))]
        #one_n_pairs = all_pairs[:len(self.features_vector)-1]
        one_n_pairs = [(i, j) for idx, (i, j) in enumerate(product([0],range(1,len(self.features_vector))))]
        
        if parallel == 1:
            #Atila: usando Ray para processamento paralelo
            #import ray
            # if n_cores == -1:
            #     n_cores = psutil.cpu_count(logical=False)
            # ray.init(num_cpus=n_cores)
            
            N_pairs_core = 20 # Testando com xxx
            if len(one_n_pairs) < N_pairs_core:
                print("Campo de busca pequeno. Reduzindo N_pairs_core para 1")
                N_pairs_core = 1
            chunks = np.array_split(one_n_pairs, len(one_n_pairs)//N_pairs_core) # Vamos dividir de modo que N_pairs_core buscas sejam feitas em cada core (1 busca é rapido demais)  
            #chunks = np.array_split(one_n_pairs, len(one_n_pairs)) # Vamos dividir de modo que cada similaridade rode em um core
            remaining_ids = []
            if n_cores > len(chunks):  #Verificando se por acaso temos mais cores do que tarefas. 
                n_proc = len(chunks)  # Nesse caso vamos disparar menos processos
            else:
                n_proc = n_cores
            print("Paralelo com Ray. Vamos usar %d cores."%n_proc)
            progressbar = Bar('Running one to %d parallel comparisons between query and reference song'%len(chunks), 
                            max=len(chunks), 
                            suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            for k in range(0,n_proc):       # loop sem wait, para iniciar processos
                remaining_ids.append(self.similarity_ray.remote(self,chunks[k]))  # Vamos disparar tantos processos quantos cores tenhamos
            #print("Todos os cores iniciados e executando. Vamos ao wait.")
            for k in range(n_proc, len(chunks)): # loop com wait, para seguir e tratar demais tarefas
                ready_ids, remaining_ids = ray.wait(remaining_ids)  #Esperando algum core terminar

                for smat in ray.get(ready_ids):
                    for ij in smat.keys():
                        i = ij[0]
                        j = ij[1]
                        score = smat[ij]
                        for sk in score.keys():
                            self.Ds[sk][i, j] = score[sk]
                    progressbar.next()
                #print("Objeto OID: %s, k: %d pronto para a proxima."%(ready_ids,k))
                remaining_ids.append(self.similarity_ray.remote(self,chunks[k]))  #Vagou. Vamos enviar proxima tarefa
            
            for smat in ray.get(remaining_ids):
                for ij in smat.keys():
                    i = ij[0]
                    j = ij[1]
                    score = smat[ij]            
                    for sk in score.keys():
                        self.Ds[sk][i, j] = score[sk]
                progressbar.next()
            ray.shutdown()
            progressbar.finish()
            #Atila: fim uso Ray.
            
            self.get_all_clique_ids() # Since nothing has been cached
        else:
            progressbar = Bar('Running one to %d sequential comparisons between query and reference song'%len(one_n_pairs), 
                            max=len(one_n_pairs), 
                            suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            for idx, (i, j) in enumerate(one_n_pairs):
                self.similarity(np.array([[i, j]]))
                if idx % 100 == 0:
                    print((i, j))
                progressbar.next()
            progressbar.finish()
        return(self.Ds)
   
    
    def all_pairwise(self, parallel=0, n_cores=12, symmetric=False, precomputed=False):
        """
        Do all pairwise comparisons between songs, with code that is 
        amenable to parallelizations.
        In the serial case where features are cached, many algorithms will go
        slowly at the beginning but then speed up once the features for all
        songs have been computed
        Parameters
        ----------
        parallel: int
            If 0, run serial.  If 1, run parallel
        n_cores: int
            Number of cores to use in a parallel scenario. If -1, all available cores.
        symmetric: boolean
            Whether comparisons between pairs of songs are symmetric.  If so, the
            computation can be halved
        precomputed: boolean
            Whether all pairs have already been precomputed, in which case we just
            want to print the result statistics
        """
        from itertools import combinations, permutations
        h5filename = "%s_Ds.h5" % self.get_cacheprefix()
        if precomputed:
            self.Ds = dd.io.load(h5filename)
            self.get_all_clique_ids()
        else:
            if symmetric:
                all_pairs = [(i, j) for idx, (i, j) in enumerate(combinations(range(len(self.filepaths)), 2))]
            else:
                all_pairs = [(i, j) for idx, (i, j) in enumerate(permutations(range(len(self.filepaths)), 2))]
            #chunks = np.array_split(all_pairs, 45) #??? Por que 45 ???
            #chunks = np.array_split(all_pairs, n_cores)

            if parallel == 1:
                #from joblib import Parallel, delayed
                #Parallel(n_jobs=n_cores, verbose=1)(
                #    delayed(self.similarity)(chunks[i]) for i in range(len(chunks)))
                    
                #Atila: usando Ray para processamento paralelo
                #import ray

                # if n_cores == -1:
                #     n_cores = psutil.cpu_count(logical=False)
                # ray.init(num_cpus=n_cores)

                N_pairs_core = 20 # Testando com xxx
                chunks = np.array_split(all_pairs, len(all_pairs)//N_pairs_core) # Vamos dividir de modo que N_pairs_core buscas sejam feitas em cada core (1 busca é rapido demais)
                #print("Trabalhando com %d pares por core - total de chunks = %d"%(N_pairs_core, len(chunks)))
                remaining_ids = []
                if n_cores > len(chunks):  #Verificando se por acaso temos mais cores do que tarefas. 
                    n_proc = len(chunks)  # Nesse caso vamos disparar menos processos
                else:
                    n_proc = n_cores
                    
                print("Paralelo com Ray. Vamos usar %d cores."%n_proc)
                progressbar = Bar('Running pairwise between all combinations of query and reference song, using %d CPUs'%n_proc, 
                                max=len(chunks) + len(self.Ds) , 
                                suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
                for k in range(0,n_proc):       # loop sem wait, para iniciar processos
                    remaining_ids.append(self.similarity_ray.remote(self,chunks[k]))  # Vamos disparar tantos processos quantos cores tenhamos
                    #remaining_ids.append(self.similarity_ray.remote(self,all_pairs[k]))  # Vamos disparar tantos processos quantos cores tenhamos
                #print("Todos os cores iniciados e executando. Vamos ao wait.")
                for k in range(n_proc, len(chunks)): # loop com wait, para seguir e tratar demais tarefas
                    ready_ids, remaining_ids = ray.wait(remaining_ids)  #Esperando algum core terminar
                    for smat in ray.get(ready_ids):
                        for ij in smat.keys():
                            i = ij[0]
                            j = ij[1]
                            score = smat[ij]
                            for sk in score.keys():
                                self.Ds[sk][i, j] = score[sk]
                        progressbar.next()
                    #print("Objeto OID: %s, k: %d pronto para a proxima."%(ready_ids,k))
                    remaining_ids.append(self.similarity_ray.remote(self,chunks[k]))  #Vagou. Vamos enviar proximo chunk
                    #remaining_ids.append(self.similarity_ray.remote(self,all_pairs[k]))  #Vagou. Vamos enviar proxima tarefa
                
                for smat in ray.get(remaining_ids):
                    for ij in smat.keys():
                        i = ij[0]
                        j = ij[1]
                        score = smat[ij]
                        for sk in score.keys():
                            self.Ds[sk][i, j] = score[sk]
                    progressbar.next()
                ray.shutdown()
                #progressbar.finish()
                #Atila: fim uso Ray.
                    
                self.get_all_clique_ids() # Since nothing has been cached
            else:
                progressbar = Bar('Running pairwise between all combinations of query and reference song', 
                                max=len(all_pairs) + len(self.Ds) , 
                                suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
                for idx, (i, j) in enumerate(all_pairs):
                    self.similarity(np.array([[i, j]]))
                    if idx % 100 == 0:
                        print((i, j))
                    progressbar.next()
            if symmetric:
                for similarity_type in self.Ds:
                    self.Ds[similarity_type] += self.Ds[similarity_type].T
                    progressbar.next()
            progressbar.finish()
            dd.io.save(h5filename, self.Ds)    

    def cleanup_memmap(self):
        """
        Remove all memmap variables for song-level similarity matrices
        """
        import shutil
        try:
            for s in self.Ds:
                shutil.rmtree('%s_%s_dmat'%(self.get_cacheprefix(), s))
        except:
            print('Could not clean-up automatically.')
    
    def getEvalStatistics(self, similarity_type, topsidx=[1, 10, 100, 1000]):
        """
        Compute MR, MRR, MAP, Median Rank, and Top X using
        a particular similarity measure
        Parameters
        ----------
        similarity_type: string
            The similarity measure to use
        """
        from itertools import chain
        D = np.array(self.Ds[similarity_type], dtype=np.float32)
        #print(D)  #Atila para debug
        N = D.shape[0]
        # Step 1: Re-sort indices of D so that
        # cover cliques are contiguous
        cliques = [list(self.cliques[s]) for s in self.cliques]
        Ks = np.array([len(c) for c in cliques]) # Length of each clique
        # Sort cliques in descending order of number
        idx = np.argsort(-Ks)
        Ks = Ks[idx]
        cliques = [cliques[i] for i in idx]
        # Unroll array of cliques and put distance matrix in
        # contiguous order
        idx = np.array(list(chain(*cliques)), dtype=int)
        D = D[idx, :]
        D = D[:, idx]
        
        # Step 2: Compute MR, MRR, MAP, and Median Rank
        # Fill diagonal with -infinity to exclude song from comparison with self
        np.fill_diagonal(D, -np.inf)
        idx = np.argsort(-D, 1) # Sort row by row in descending order of score
        ranks = np.nan*np.ones(N)
        startidx = 0
        kidx = 0
        AllMap = np.nan*np.ones(N)
        for i in range(N):
            if i >= startidx + Ks[kidx]:
                startidx += Ks[kidx]
                kidx += 1
                if Ks[kidx] < 2:
                    # We're done once we get to a clique with less than 2
                    # since cliques are sorted in descending order
                    break
            iranks = []
            for k in range(N):
                diff = idx[i, k] - startidx
                if diff >= 0 and diff < Ks[kidx]:
                    iranks.append(k+1)
            iranks = iranks[0:-1] # Exclude the song itself, which comes last
            if len(iranks) == 0:
                warnings.warn("Recalling 0 songs for clique of size %i at song index %i"%(Ks[kidx], i))
                break
            # For MR, MRR, and MDR, use first song in clique
            ranks[i] = iranks[0] 
            # For MAP, use all ranks
            P = np.array([float(j)/float(r) for (j, r) in \
                            zip(range(1, Ks[kidx]), iranks)])
            AllMap[i] = np.mean(P)
        MAP = np.nanmean(AllMap)
        ranks = ranks[np.isnan(ranks) == 0]
        print(ranks)
        MR = np.mean(ranks)
        MRR = 1.0/N*(np.sum(1.0/ranks))
        MDR = np.median(ranks)
        print("%s %s STATS\n-------------------------\nMR = %.3g\nMRR = %.3g\nMDR = %.3g\nMAP = %.3g"
              % (self.name, similarity_type, MR, MRR, MDR, MAP))
        tops = np.zeros(len(topsidx))
        for i in range(len(tops)):
            tops[i] = np.sum(ranks <= topsidx[i])
            print("Top-%i: %i"%(topsidx[i], tops[i]))
        
        # Output to CSV file
        resultsfile = "results_%s_%s.csv" % (self.shortname, self.name)
        if not os.path.exists(resultsfile):
            fout = open(resultsfile, "w")
            fout.write("name, MR, MRR, MDR, MAP")
            for t in topsidx:
                fout.write(",Top-%i"%t)
            fout.write("\n")
        fout = open(resultsfile, "a")
        fout.write("%s_%s,"%(self.name, similarity_type))
        fout.write("%.3g, %.3g, %.3g, %.3g"%(MR, MRR, MDR, MAP))
        for t in tops:
            fout.write(", %.3g"%t)
        fout.write("\n")
        fout.close()
        return MR, MRR, MDR, MAP, tops
