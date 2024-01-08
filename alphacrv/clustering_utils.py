"""
Clustering functions
"""

from pathlib import Path
import shutil
import subprocess
import json
import pickle
from typing import Union, Tuple, List, Dict, Set
from subprocess import CalledProcessError
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import numpy as np
from itertools import combinations
import multiprocessing
import biskit as b
import os

import pandas as pd

import logging
logging.getLogger().setLevel(logging.INFO)

from alphacrv.run_usalign import calculate_tmscore


def run_sequence_clustering(destination:Path, sequences:List[SeqRecord],
                        results_prefix:str="clusterRes", temp_dir:str="tmp",
                        min_seq_id:float=0.3, coverage:float=0.8, cov_mode:int=2,
                        sensitivity:int=7):
    """
    Run mmseqs easy-cluster to cluster the sequences by sequence identity

    Args:
        destination (Path): Parent destination path
        sequences (List[SeqRecord]): List of SeqRecords with the sequences
        results_prefix (str, optional): Prefix for the result files. Defaults to "clusterRes".
        temp_dir (str, optional): Defaults to "tmp".
        min_seq_id (float, optional): Defaults to 0.3.
        coverage (float, optional): Defaults to 0.8.
        cov_mode (int, optional): Defaults to 2.
        sensitivity (int, optional): Defaults to 7.
    """
    
    out_seqcluster = destination / "seqclusters"
    out_seqcluster.mkdir(exist_ok=True)
    
    sequences_file = out_seqcluster / "sequences.fasta"
    SeqIO.write(sequences, sequences_file, "fasta")

    seqclusters_tsv = out_seqcluster / "clusterRes_cluster.tsv"

    if not seqclusters_tsv.exists():

        logging.info("Running sequence clustering...")
        command = (f"mmseqs easy-cluster {sequences_file.resolve()} {results_prefix} {temp_dir} "
                f"--min-seq-id {min_seq_id} -c {coverage} --cov-mode {cov_mode} "
                f"-s {sensitivity}").split()
        try:
            p = subprocess.run(command, cwd=out_seqcluster, capture_output=True)
            p.check_returncode()
        except CalledProcessError as e:
            fail(p, "mmseqs", command, e)
        
    else:
        logging.info("Sequence clustering output already exists.")
        
    logging.info("Processing output...")
    seqclusters = pd.read_table(seqclusters_tsv, header=None, names=["rep", "member"])

    return seqclusters


def get_top_pdbs(models_dir:Path, destination:Path):
    """
    Copy the top ranked models for each complex to a new directory.

    Args:
        models_dir (Path): Path to the directory with the models
        destination (Path): Path to the directory where the pdbs will be copied
    """
    
    pdbs_dir = destination / "all_pdbs"
    pdbs_dir.mkdir(exist_ok=True)
    
    count=0
    for d in models_dir.iterdir():
        if d.is_dir():
            top_pdb = d / "ranked_0.pdb"
            if top_pdb.exists():
                count += 1
                # Copy top pdb to new destination
                new_name = pdbs_dir / (d.name + ".pdb")
                shutil.copy(top_pdb, new_name)
    
    logging.info(f"Copied {count} models to {pdbs_dir}.")
    
    return pdbs_dir


def run_structure_clustering(destination:Path, top_models_dir:Union[Path, None],
                    models_dir:Path=None, results_prefix:str="clusterRes",
                    temp_dir:str="tmp", coverage:float=0.8, cov_mode:int=2,
                    evalue:float=0.01) -> Tuple[pd.DataFrame, Path]:
    """
    Run foldseek easy-cluster to cluster the models by structure

    Args:
        destination (Path): Parent destination path
        model_names (List[str]): List of model names to be clustered
        top_models_dir (Path): Path to the directory with the PDBs to cluster
        models_dir (Path): Path to the directory with the models from AlphaFold
        results_prefix (str, optional): Defaults to "clusterRes".
        temp_dir (str, optional): Defaults to "tmp".
        coverage (float, optional): Defaults to 0.8.
        cov_mode (int, optional): Defaults to 2.
        evalue (float, optional): Defaults to 0.01.
    """
    
    out_strcluster = destination / "strclusters"
    out_strcluster.mkdir(exist_ok=True)

    if top_models_dir is None:
        logging.info("Copying models to new directory...")
        pdbs_dir = get_top_pdbs(models_dir, destination)
    else:
        pdbs_dir = top_models_dir
    
    strclusters_tsv = out_strcluster / "clusterRes_cluster.tsv"

    if not strclusters_tsv.exists():

        logging.info("Running structural clustering...")
        command = (f"foldseek easy-cluster {pdbs_dir.resolve()} {results_prefix} {temp_dir} "
                f"-c {coverage} --cov-mode {cov_mode} -e {evalue}").split()
        
        try:
            p = subprocess.run(command, cwd=out_strcluster, capture_output=True)
            p.check_returncode()
        except CalledProcessError as e:
            fail(p, "foldseek", command, e)
    
    else:
        logging.info("Structure clustering output already exists.")
    
    logging.info("Processing output...")
    strclusters = pd.read_table(strclusters_tsv, header=None, names=["rep", "member"])

    return strclusters, pdbs_dir


def fail(process:subprocess.CompletedProcess, program:str, args:list,
         error:CalledProcessError):
    """
    Generates the error message and raises the corresponding error if the
    program fails.
    """
    error_string = \
        f"\n{program} EXECUTION FAILED.\n"+\
        f"Command: {' '.join(args)}\n"

    error_string += \
        f"Returncode: {process.returncode}\n"+\
        f"STDERR: \n"+\
        process.stderr.decode("utf-8")

    logging.error(error_string)
    
    raise error


def get_scores(models_dir:Path) -> pd.DataFrame:
    """
    Function to obtain iptm and iptm+ptm scores from the top ranked models.

    Args:
        models_dir (Path): Directory with AlphaFold models

    Returns:
        pd.DataFrame: DataFrame with the scores
    """
    
    scores = {}
    for d in models_dir.iterdir():
        ranking_file = d / "ranking_debug.json"
        if ranking_file.exists():
            with open(d / "ranking_debug.json") as f:
                qscores = json.load(f)
                rank0 = qscores['order'][0]
                iptm_ptm = qscores['iptm+ptm'][rank0]
                
            with open(d / f"result_{rank0}.pkl", "rb") as f:
                result = pickle.load(f)
                iptm = float(result['iptm'])
            
            scores[d.name] = {"iptm": iptm, "iptm+ptm": iptm_ptm} 

    scores = pd.DataFrame.from_dict(scores, orient="index")
    scores = scores.reset_index().rename(columns={"index": "complex"})
    
    logging.info(f"Found {len(scores)} model directories with quality scores.")

    scores = scores.sort_values(by=['iptm', 'iptm+ptm'], ignore_index=True,
                                ascending=False)

    return scores


def get_iptm_threshold(scores:pd.DataFrame, iptm_threshold:float) -> float:
    """
    Function to obtain the iptm threshold to use for the clustering. Loop to
    show the user how many models will be used, and ask either for confirmation
    or a new threshold.

    Args:
        scores (pd.DataFrame): DataFrame with the iptm and iptm+ptm scores
        iptm_threshold (float, optional): Threshold for iptm score. Defaults to 0.5.

    Returns:
        float: iptm threshold
    """
    
    while True:
        select = scores[scores.iptm >= iptm_threshold]
        new_threshold = input(f"Will select {len(select)} models with iptm >= "
                              f"{iptm_threshold}. "
                               "Press enter to continue, or enter a new threshold: ")
        if new_threshold:
            if float(new_threshold) < 0 or float(new_threshold) > 1:
                print("The iptm threshold must be between 0 and 1.")
            else:
                iptm_threshold = float(new_threshold)
        else:
            break
    
    return iptm_threshold


################ Functions for merging clusters ################

def get_topcluster_members(clusters:pd.DataFrame, min_count:int=2) -> Dict[str, Set[str]]:
    """
    Obtain the top clusters in dictionary format, with the cluster representative
    as the key and the members as the values in a set.

    Args:
        clusters (pd.DataFrame): _description_
        min_count (int, optional): _description_. Defaults to 2.

    Returns:
        dict: _description_
    
    E.g.
    {'Q5N7Y5': {'Q5N7Y5', 'Q69XA8', 'Q6K7U3'},
     'Q2QN41': {'Q2QN41', 'Q8W0W4'},
     'A0A0P0XNZ0': {'A0A0P0XNZ0', 'A0A0P0XQ16'}}
    """
    # Get the member count
    member_counts = clusters.rep.value_counts()
    # Get the clusters with more than one member
    top_clusters = member_counts[member_counts >= min_count].index
    # Get the members of the top clusters
    top_clusters_members = {r:set(clusters[clusters.rep==r].member.tolist()) \
                           for r in top_clusters}
    
    return top_clusters_members


def merge_dict_values(unmerged_dict:Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Take a dictionary with sets as values, and merge the sets that have elements
    in common.

    Args:
        d (Dict[str, Set[str]]): _description_

    Returns:
        Dict[str, Set[str]]: _description_
    """
    
    joint_dict = {}
    while unmerged_dict:
        rep, members = unmerged_dict.popitem()

        if members is None:
            continue

        merged_members = members
        keys_to_merge = []
        for rep2, members2 in unmerged_dict.items():
            if members2 is None:
                continue
            if not merged_members.isdisjoint(members2):
                merged_members = merged_members | members2
                keys_to_merge.append(rep2)

        # Erase the values of the merged clusters
        for rep2 in keys_to_merge:
            unmerged_dict[rep2] = None
        
        # Check in the joint dictionary if the merged cluster has elements in
        # common with other clusters
        keys_to_merge = []
        for rep2, members2 in joint_dict.items():
            if not merged_members.isdisjoint(members2):
                merged_members = merged_members | members2
                keys_to_merge.append(rep2)
        
        # Erase the old values from the joint dictionary
        for rep2 in keys_to_merge:
            del joint_dict[rep2]

        joint_dict[rep] = merged_members

    return joint_dict



def joint_cluster(seqclusters:pd.DataFrame, strclusters:pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Function to merge the clusters from the structure and sequence clustering.
    The top clusters from the structure clustering are taken as the base, and
    the top clusters from the sequence clustering are merged into them. If a
    sequence cluster has elements in common with more than one structure
    cluster, they are merged into the same cluster.

    Args:
        seqclusters (pd.DataFrame): sequence clusters
        strclusters (pd.DataFrame): structure clusters

    Returns:
        Dict[str, Set[str]]: Dictionary with the merged clusters. The keys are
            the cluster representatives, and the values are the members of the
            cluster. 
    """
    
    top_seqclusters_members = get_topcluster_members(seqclusters)
    top_strclusters_members = get_topcluster_members(strclusters)
    
    # Iterate over the top structure clusters, and see if they have elements in
    # common with the top sequence clusters. If they do, join them.
    joint_clusters = {}
    merged_seqclusters = []
    for rep, members in top_strclusters_members.items():
        merged_members = members
        for rep2, members2 in top_seqclusters_members.items():
            if not merged_members.isdisjoint(members2):
                merged_members = merged_members | members2
                merged_seqclusters.append(rep2)
        
        joint_clusters[rep] = merged_members

    # Merge the missing sequence clusters
    missing_seqclusters = set(top_seqclusters_members.keys()) - set(merged_seqclusters)
    if missing_seqclusters:
        for rep in missing_seqclusters:
            joint_clusters[rep] = top_seqclusters_members[rep]
            
    # Merge joint clusters further in cases where they have elements in common
    joint_clusters2 = merge_dict_values(joint_clusters)
    
    return joint_clusters2


def joint_clusters_df(seqclusters:pd.DataFrame, strclusters:pd.DataFrame
                          ) -> pd.DataFrame:
    """
    Get the DataFrame with the joint clusters from the clusters of the structures
    and the clusters of the sequences.
    """
    
    joint_clusters = joint_cluster(seqclusters, strclusters)
    
    # Modify old columns and set `complex` as the index
    strclusters = strclusters.rename(columns={'rep':'str_rep'}).set_index('complex')
    seqclusters = seqclusters.rename(columns={'rep':'seq_rep'}).set_index('complex')

    # Add the `seq_rep` column to the `strclusters` dataframe
    strclusters['seq_rep'] = seqclusters.seq_rep
    
    # Initialize a new column `merged_rep`
    strclusters['merged_rep'] = None
    
    # Iterate over joint_clusters and change the values of the `merged_rep` column
    # in the `strclusters` dataframe
    for rep, members in joint_clusters.items():
        strclusters.loc[strclusters.member.isin(members), 'merged_rep'] = rep

    strclusters.fillna(np.nan, inplace=True)
    
    return strclusters


################ Functions to align the clusters ################

def align_all(clusters:pd.DataFrame,
              pdbs_dir: Path, cpus:int=1) -> pd.DataFrame:
    """
    Align all vs all the members of every cluster

    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        pdbs_dir (Path): Path to the directory with the models
        cpus (int, optional): Number of CPUs to use. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame with the alignment scores
    """
    # Filter out clusters with no merged_rep
    clusters_names = clusters[clusters.merged_rep.notna()].merged_rep.unique()
    len_clusters = len(clusters_names)
    
    aligned_dfs = []
    for i, cluster in enumerate(clusters_names):
        
        logging.info(f"Aligning cluster {i+1} of {len_clusters}...")
        
        members = list(clusters[clusters.merged_rep == cluster].member.values)
        
        logging.info(f"{len(members)} members.")
        
        member_combinations = list(combinations(members, 2))
        member_paths = [(pdbs_dir / (m1 + ".pdb"), pdbs_dir / (m2 + ".pdb")) \
                                        for m1, m2 in member_combinations]

        with multiprocessing.Pool(min(cpus, len(member_combinations))) as pool:
            
            results = pool.starmap(calculate_tmscore, member_paths)
        
        
        aligned_scores = [(cluster, m1, m2, tmscore_m1, tmscore_m2, aligned_length, rmsd) \
                            for (m1, m2), (aligned_length, rmsd, tmscore_m1, tmscore_m2) \
                            in zip(member_combinations, results)]

        # Make dataframe
        columns = ['cluster', 'ref', 'member', 'tmscore_ref', 'tmscore_m',
                'aligned_length', 'rmsd']
        aligned_df = pd.DataFrame(aligned_scores, columns=columns)
        aligned_dfs.append(aligned_df)

    return pd.concat(aligned_dfs).reset_index(drop=True)


def medians_alignments(alignment_scores:pd.DataFrame,
                       clusters:pd.DataFrame) -> pd.DataFrame:
    """
    Get the median alignment scores for each cluster

    Args:
        alignment_scores (pd.DataFrame): DataFrame with the alignment scores
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
                                 members
        pdbs_dir (Path): Path to the directory with the models
    Returns:
        pd.DataFrame: DataFrame with the median alignment scores
    """
    
    # Concatenate the scores for the reference and the member
    columns = ['cluster', 'ref', 'tmscore_ref', 'rmsd', 'aligned_length']
    scores1 = alignment_scores[columns].copy().rename(
                                columns={'ref': 'member', 'tmscore_ref': 'tmscore'})

    columns = ['cluster', 'member', 'tmscore_m', 'rmsd', 'aligned_length']
    scores2 = alignment_scores[columns].copy().rename(
                                columns={'tmscore_m': 'tmscore'})

    all_scores = pd.concat([scores1, scores2])
    
    # Create a column for cluster size
    all_scores['cluster_size'] = all_scores.cluster.map(
                                             clusters.merged_rep.value_counts())
    
    # Calculate the median values for each member of each cluster
    median_scores = all_scores.groupby(['cluster', 'member']).median().reset_index()

    return median_scores


def add_binder_fraction(median_scores:pd.DataFrame, pdbs_dir:Path) -> pd.DataFrame:
    """
    Add the fraction of the binder in the alignment to the median scores
    """

    # Create column for the fraction of the binder in the alignment
    median_scores['fraction_binder'] = 0.0
    for i in median_scores.index:
        m = median_scores.loc[i, 'member']
        m_pdb = pdbs_dir / (m + ".pdb")
        
        model = b.PDBModel(os.fspath(m_pdb))
        length_bait = len(model.takeChains([0]).sequence())
        length_binder = len(model.takeChains([1]).sequence())
        
        aligned_length = median_scores.loc[i, 'aligned_length']
        median_scores.loc[i, 'fraction_binder'] = ((aligned_length - length_bait)
                                                                / length_binder)

    return median_scores


################ Functions for clustering clusters ################

def get_top_clusters(median_scores:pd.DataFrame,
                     min_members:int,
                     min_tmscore:float=0.2,
                     min_fraction_binder:float=0.2,
                     max_rmsd:float=15.0) -> Tuple[list, pd.DataFrame]:
    """
    Get the top clusters based on the median alignment score and fraction of
    the binder

    Args:
        median_scores (pd.DataFrame): DataFrame with the median alignment scores
        min_members (int): Minimum number of members for a cluster to be
            considered
        min_tmscore (float): Minimum median TM-score for a cluster to be
            considered
        min_fraction_binder (float): Minimum fraction of the binder for a
            cluster to be considered

    Returns:
        list of str: List with the top clusters
    """

    # Select the clusters based on all the criteria
    select = ((median_scores.cluster_size >= min_members) & \
              (median_scores.tmscore >= min_tmscore) & \
              (median_scores.fraction_binder >= min_fraction_binder) & \
              (median_scores.rmsd <= max_rmsd))
    median_scores_filtered = median_scores[select]

    clusters = list(median_scores_filtered.cluster.unique())
    
    return clusters, median_scores_filtered


def copy_pdbs(clusters:pd.DataFrame, pdbs_dir:Path, destination:Path,
              topclusters:list) -> None:
    """
    Copy the top pdbs from the top clusters to a new directory
    
    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        pdbs_dir (Path): Path to the directory with the models
        destination (Path): Path to the directory where the pdbs will be copied
        topclusters (list): List with the top clusters
    """
    
    for cluster in topclusters:
        # Get cluster members
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster_{cluster}"
        cluster_dir.mkdir(exist_ok=True)
        for member in members:
            pdb_name = pdbs_dir / (member + ".pdb")
            assert pdb_name.exists(), f"{pdb_name} doesn't exist"
            new_name = cluster_dir / (member + ".pdb")
            shutil.copy(pdb_name, new_name)


def cluster_clusters(destination:Path, topclusters:list):
    """
    For each cluster:
    1. Do structural clustering to identify the "consensus" structure of the cluster
    2. Use us-align to align all the members of the cluster to the consensus structure
    3. Save the alignment scores in a DataFrame
    
    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        destination (Path): Path to the directory with the pdbs for each cluster
        topclusters (list): List with the top clusters
    """
    clustered_clusters = []
    for cluster in topclusters:

        logging.info(f"Clustering {cluster}")

        cluster_dir = destination / f"cluster_{cluster}"
        cluster_clusters_dir = destination / f"cluster_{cluster}_clusters"
        if not cluster_clusters_dir.exists():
            cluster_clusters_dir.mkdir()
        
        # Do structural clustering on the merged chains
        strclusters, pdbs_dir = run_structure_clustering(
                                    destination=cluster_clusters_dir,
                                    top_models_dir=cluster_dir)
        
        # Get only the clusters for the structures of the binder
        strclusters = strclusters[strclusters.member.str.endswith('_B')]
        # Remove the '.pdb_B' suffix from the members' names
        strclusters['member'] = strclusters['member'].str.split('.pdb').str[0]
        strclusters['cluster'] = cluster
        
        strclusters.rename(columns={'rep':'subcluster_rep'}, inplace=True)
        
        clustered_clusters.append(strclusters)
    
    return pd.concat(clustered_clusters)
