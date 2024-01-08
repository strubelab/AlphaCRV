# AlphaCRV
---

AlphaCRV is a python package that helps identify correct interactors in a one-against-many AlphaFold-Multimer screen by clustering, ranking, and visualizing conserved binding topologies, based on protein sequence and fold. 

# Installation

## Install in a conda environment

The latest version of AlphaCRV can be installed in a linux system with the following commands (requires `conda`):

```bash
## Install the alphacrv library and dependencies in an virtual environment
git clone https://github.com/strubelab/AlphaCRV
cd AlphaCRV
conda env create --file environment.yml --prefix ./env
conda activate ./env
pip install -e .

## Install the biskit library
git clone https://github.com/graik/biskit.git
pip install -e ./biskit

## Install US-align
mkdir ./usalign; cd ./usalign
wget https://zhanggroup.org/US-align/bin/module/USalign.cpp
g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp  # Tested with with g++ 6.4.0
ln USalign ../env/bin/
```

# Usage

AlphaCRV can be run from the command line in two stages:

## 1. Clustering

The first stage performs the following actions to cluster structures:

1. Obtains quality scores for the models, and selects the models with an interface predicted TM-score (ipTM) equal to or greater than the set threshold (`--iptm_threshold`). This is done to keep the models that have a high quality interface prediction.
2. Uses the [biskit](https://github.com/graik/biskit) python library to trim the PDB models and sequences based on a set PAE threshold (`--pae_threshold`). This is done to only keep the residues/domains that are confidently modeled.
3. Runs [`mmseqs2 easy-cluster`](https://github.com/soedinglab/MMseqs2) to cluster the binder sequences, and [`foldseek easy-cluster`](https://github.com/steineggerlab/foldseek) to cluster the binder structures. The sequence and structure clusters are then combined to create the final clusters.
4. The elements of each cluster are then aligned using [`USalign`](https://zhanggroup.org/US-align/), to find the best representative structure for each cluster. These representatives can then be used to rank the clusters.

The clustering stage can be run using the following command:

```bash
alphacrv-cluster \
  --bait /path/to/bait_sequence.fasta \
  --binders /path/to/binder_sequences.fasta \
  --models_dir /path/to/alphafold_multimer_models \
  --destination /path/to/results_dir \
  --pae_threshold 10 \
  --cpus 8
```

- `--bait` is the path to the bait sequence in FASTA format. This file should contain only one sequence.
- `--binders` is the path to the binder sequences in FASTA format.
- `--models_dir` is the path to the directory containing the AlphaFold-Multimer models. Each directory inside should contain the AlphaFold-Multimer outputs such as:
    - `ranking_debug.json` for the ranking of the models.
    - `ranked_0.pdb` for the top ranked model.
    - `result_model_4_multimer_v3_pred_0.pkl` for the full output of the top model

    For an example of the directory structure, see the [example directory](example/).
- `--destination` is the path to the directory where the results will be saved.
- `--pae_threshold` is the PAE threshold used to trim the models and sequences. For each PDB, the program will iteratively increase this threshold by 5 units if the length of the region below the given threshold is less than 20 residues.
- `--cpus` is the number of CPUs to use for the alignment step with `USalign`.

To see a detailed description of the parameters, run `alphacrv-cluster --help`.

This command should create the following files in the `--destination` directory:

- `binder_regions.csv`: A CSV file containing the trimmed regions of each binder.
- `trimmed_binders.fasta`: A FASTA file containing the trimmed sequences of each binder.
- `pdbs_trimmed/` : A directory containing the trimmed PDB complexes.
- `merged_clusters/`
    - `merged_clusters.csv`: Merged clusters.
    - `alignment_scores.csv`: Alignment scores of the elements of each cluster.
    - `median_scores.csv`: Median alignment scores of the elements of each cluster.
- `seqclusters/` : Contains the results of the `mmseqs2 easy-cluster` run on the binder sequences.
- `structclusters/` : Contains the results of the `foldseek easy-cluster` run on the binder structures.


## 2. Make PyMol sessions for visualization of the best clusters, and create subclusters of each cluster

This stage performs the following actions:

1. Obtain the top clusters from the merged clusters created in the previous step, based on several ranking criteria (see example below).
2. Copy the PDBs from each top cluster to a new directory, and create a PyMol session with the members of each cluster.
3. Use [`foldseek easy-cluster`](https://github.com/steineggerlab/foldseek) to create subclusters of each top cluster. This might be useful to identify alternative binding modes of the same binder, or multiple domains involved in an interaction.

The second stage can be run using the following command:

```bash
alphacrv-rank \
  --clusters_dir /path/to/results_dir \
  --min_members 5 \
  --min_tmscore 0.2 \
  --max_rmsd 15 \
  --cluster_clusters
```

- `--clusters_dir` is the path to the directory containing the results from the clustering stage.
- `--min_members` is the minimum number of members a cluster should have to be considered.
- `--min_tmscore` is the minimum median TM-score of the cluster representative againts all other cluster members. This is used to filter out clusters with poor alignments.
- `--max_rmsd` is the maximum RMSD of the cluster representative againts all other cluster members. This is used to filter out clusters with poor alignments.
- `--cluster_clusters` is a flag to indicate whether to create subclusters of each top cluster.

To see a detailed description of the parameters, run `alphacrv-rank --help`.

This command should create the following files in the `--clusters_dir` / `merged_clusters` directory:

- `clustered_clusters.csv`: Contains the subclusters for each of the top clusters.
- `cluster_<cluster_representative>/`: Contains the PDBs of each cluster, and a PyMol session with the cluster members.
- `cluster_<cluster_representative>_clusters/`: Contains the results of the `foldseek easy-cluster` run on the cluster members.


# Example

To see AlphaCRV in action, please refer to the Jupyter notebooks in the [examples directory](./examples/).

# Citation

If you use AlphaCRV in your research, please cite the following paper:

*Submitted to Bioinformatics*
