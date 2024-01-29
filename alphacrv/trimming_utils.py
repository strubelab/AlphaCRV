"""
Functions to extract the regions of the candidate models that have a high PAE score
against the bait
"""

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Tuple, List

import biskit as b
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd

from alphacrv.clustering_utils import get_id

logging.getLogger().setLevel(logging.INFO)


def get_pae(model_dir:Path) -> np.ndarray:
    """
    Read the results dictionary of the top model from the pickle file
    """
    with open(model_dir / 'ranking_debug.json', 'r') as f:
        model_rank = json.load(f)['order']
        
    pickle_name = model_dir / f'result_{model_rank[0]}.pkl'
    
    with open(pickle_name, 'rb') as f:
        results = pickle.load(f)

    pae = results['predicted_aligned_error']
    
    return pae


def get_border_indices(mean_pae:np.ndarray, pthresh:float
                       ) -> Tuple[int, int]:
    """
    Get the indices of the first and last appearances of a pae score below
    the threshold.

    Args:
        slength (int): length of the bait sequence (chain A)
        mean_pae (np.ndarray): array of length complex_length with the mean
                                 predicted aligned error
        pthresh (float): pae threshold

    Returns:
        Tuple[int, int]: indices of the first and last appearances of a pae
                            score below the threshold
    """

    belowthresh_indices = np.where(mean_pae <= pthresh)
    if len(belowthresh_indices[0]) == 0:
        btleft = np.nan
        btright = np.nan
    else:
        btleft = belowthresh_indices[0][0]
        btright = belowthresh_indices[0][-1]
    
    return btleft, btright


def get_lowpae_indices(pae:np.ndarray, slength:int, pthresh:float,
                       nbait:int=1) -> Tuple[int, int, float]:
    """
    Obtain the indices of the first and last appearances of a pae score below
    the threshold.

    Args:
        pae (np.ndarray): array of length complex_length X complex_length with
                          the predicted aligned error
        slength (int): length of the bait sequence (chain A)
        pthresh (float): pae threshold
        nbait (int, optional): Number of bait molecules in the complex. Defaults to 1.

    Returns:
        Tuple[int, int, float]: indices of the first and last appearances of a
                                pae score below the threshold, and the threshold
                                used
    """

    # Extract the rows corresponding to the first sequence and the columns
    # corresponding to the second sequence.
    # Then calculate the mean of each column
    mean_pae1 = np.mean(pae[:slength*nbait, slength*nbait:], axis=0)
    btleft1, btright1 = get_border_indices(mean_pae1, pthresh)

    # Extract the columns corresponding to the first sequence and the rows
    # corresponding to the second sequence.
    # Then calculate the mean of each row.
    mean_pae2 = np.mean(pae[slength*nbait:, :slength*nbait], axis=1)
    btleft2, btright2 = get_border_indices(mean_pae2, pthresh)

    if all(np.isnan([btleft1, btright1, btleft2, btright2])):
        # keep increasing the threshold until we find a region with low PAE
        minleft, maxright, pthresh = get_lowpae_indices(pae, slength, pthresh+5,
                                                        nbait)
        return minleft, maxright, pthresh
    else:
        # Find the minimum of the left indices, where one could be nan
        btleft1 = btleft1 if not np.isnan(btleft1) else np.inf
        btleft2 = btleft2 if not np.isnan(btleft2) else np.inf
        minleft = np.min([btleft1, btleft2])
        assert not np.isinf(minleft)

        # Find the maximum of the right indices, where one could be nan
        btright1 = btright1 if not np.isnan(btright1) else -np.inf
        btright2 = btright2 if not np.isnan(btright2) else -np.inf
        maxright = np.max([btright1, btright2])
        assert not np.isinf(maxright)

    # Check that the indices are in the correct order
    minleft, maxright = np.min([minleft, maxright]), np.max([minleft, maxright])

    # Check that the region is at least 10 residues long
    if maxright - minleft < 19:
        # keep increasing the threshold until we find a region of at least 20
        # residues with low PAE
        minleft, maxright, pthresh = get_lowpae_indices(pae, slength, pthresh+5,
                                                        nbait)
        return minleft, maxright, pthresh

    return int(minleft), int(maxright)+1, pthresh


def extract_pdb_region(m:b.PDBModel, leftind:int, rightind:int, destination:Path,
                       id_binder:str) -> None:
    """
    Extract the region of the second chain of the pdb that is between the left
    and right indices provided. Then write the pdb to disk.
    
    Args:
        pdb (Path): Path to the pdb file
        leftind (int): Left index of the region to be extracted
        rightind (int): Right index of the region to be extracted
        destination (Path): Path to the directory where the pdb will be written
        id_binder (str): ID of the binder. Will be used to name the trimmed pdb
    """
    
    chain_indices = list(np.arange(m.lenChains()))
    chainb = m.takeChains([chain_indices[-1]])
    chainb = chainb.takeResidues(list(np.arange(leftind, rightind)))
    
    m2 = m.takeChains(chain_indices[:-1]).concat(chainb)
    
    # Get new name to write pdb
    # uid = re.search(r'-1_(\w+)-1', pdb.parent.name).group(1)
    new_name = destination / (id_binder + ".pdb")
    
    m2.writePdb(os.fspath(new_name))


def trim_models(destination:Path, model_names: List[str],
                sequences:List[SeqRecord], models_dir:Path,
                pae_threshold:float=15.0, nbait:int=1,
                ) -> Tuple[List[SeqRecord], Path]:
    """
    Trim the models listed in model_names to the largest region with a PAE
    score below the threshold. Just take the first and last indices of the
    sequence below the threshold and extract that region.
    The following files are saved to disk:
    - Trimmed binder sequences: destination / "trimmed_binders.fasta"
    - Trimmed regions of the binders: destination / "binders_regions.csv"
    - Trimmed PDBs: destination / "pdbs_trimmed"

    Args:
        bait (Path): Fasta file with the bait sequence
        destination (Path): Destination directory to save the outputs
        model_names (List[str]): List of model names to be trimmed
        sequences (List[SeqRecord]): List of SeqRecords with the sequences to be
                                        trimmed
        models_dir (Path): Path to the directory with the models before trimming
        pae_threshold (float, optional): PAE threshold to trim models.
                                            Defaults to 15.0.
        nbait (int, optional): Number of bait molecules in the complex. Defaults to 1.

    Returns:
        Tuple[List[SeqRecord], Path]: List of trimmed binders and path to the
                                        directory with the trimmed PDBs
    """
    
    sequences = {get_id(s.id): s for s in sequences}
    
    pdbs_dir = destination / "pdbs_trimmed"
    if not pdbs_dir.exists():
        pdbs_dir.mkdir()
    
    trimmed_binders = []
    binders_regions = []
    
    count = 0
    
    for m in model_names:
        d = models_dir / m
        top_pdb = d / "ranked_0.pdb"
        if top_pdb.exists():
            count += 1
            
            # Get the length of the bait directly from the first chain of the pdb
            m = b.PDBModel(os.fspath(top_pdb))
            len_bait = m.takeChains([0]).lenResidues()
            
            pae = get_pae(d)
            
            # Get the indices of the region with low PAE
            leftind, rightind, pthresh = get_lowpae_indices(pae,
                                                        len_bait,
                                                        pae_threshold, nbait)
            
            # Save the sequence of the trimmed region
            id_binder = re.search(r'(-\d)?_(\w+)(-\d)?', m).group(2)
            seq_trimmed = sequences[id_binder][leftind:rightind]
            trimmed_binders.append(seq_trimmed)
            
            # Read pdb and extract the region
            extract_pdb_region(m, leftind, rightind, pdbs_dir, id_binder)
            
            # Save the region indices
            binders_regions.append((seq_trimmed.id, pthresh, leftind,
                                        rightind))
        else:
            logging.warning(f"Could not find top ranked pdb for {m}")
    
    logging.info(f"Processed {count} complexes.")
    
    logging.info("Writing trimmed sequences to fasta file.")
    SeqIO.write(trimmed_binders, destination / "trimmed_binders.fasta",
                'fasta')
    
    # Write the regions of the candidates to a file
    pd.DataFrame(binders_regions,
                 columns=['id', 'pae_threshold', 'left', 'right']).to_csv(
        destination / "binders_regions.csv", index=False)
                 
    return trimmed_binders, pdbs_dir