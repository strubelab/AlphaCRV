"""
Functions to select top clusters and make pymol sessions
"""

from pathlib import Path
import re
from pymol import cmd
import pandas as pd


def make_pymol_sessions(clusters:pd.DataFrame, destination:Path,
                        topclusters:list) -> None:
    """
    Create the pymol session with the superimposition of each of the clusters

    Args:
        clusters (pd.DataFrame): DataFrame with the clustering results
        destination (Path): Path to the directory with the pdbs for each cluster
        topclusters (list): List with the top clusters
    """
    
    for cluster in topclusters:
        # Get cluster members
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster_{cluster}"

        # Load the cluster representative first
        if re.search(r'.pdb_[A-Z]$', cluster):
            chain = cluster[-1]
            cname = cluster.split('.pdb')[0]
            fname = cluster_dir / (cname + ".pdb")
            oname = f"{cname}_rep{chain}"
            cmd.load(fname, oname)
            cmd.do(f"select chain {chain} AND model {oname}")
        else:
            fname = cluster_dir / (cluster + ".pdb")
            cmd.load(fname, f"{cluster}_rep")
            cmd.do(f"select chain B AND model {cluster}_rep")
        
        # Load and align the members one by one
        for member in members:
            if not member in cluster:
                fname = cluster_dir / (member + ".pdb")
                cmd.load(fname)
                cmd.align(member, "sele")
        
        cmd.do('bg white')
        cmd.do('set ray_shadow, 0')
        cmd.do('color grey80')
        cmd.do('select chain A')
        cmd.do('color slate, sele')
        
        cmd.save(cluster_dir / "session.pse")
        cmd.do('delete all')
