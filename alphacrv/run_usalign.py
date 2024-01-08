"""
Functions to run the US-align algorithm.

Example command and output:

$ USalign Q5JL91.pdb A0A0N7KFK3.pdb -TMscore 7 -ter 1

 ********************************************************************
 * US-align (Version 20230609)                                      *
 * Universal Structure Alignment of Proteins and Nucleic Acids      *
 * Reference: C Zhang, M Shine, AM Pyle, Y Zhang. (2022) Nat Methods*
 * Please email comments and suggestions to zhang@zhanggroup.org    *
 ********************************************************************

Name of Structure_1: Q5JL91.pdb:A:B (to be superimposed onto Structure_2)
Name of Structure_2: A0A0N7KFK3.pdb:A:B
Length of Structure_1: 168 residues
Length of Structure_2: 188 residues

Aligned length= 164, RMSD=  18.48, Seq_ID=n_identical/n_aligned= 0.683
TM-score= 0.51426 (normalized by length of Structure_1: L=168, d0=4.83)
TM-score= 0.46125 (normalized by length of Structure_2: L=188, d0=5.11)
(You should use TM-score normalized by length of the reference structure)

(":" denotes residue pairs of d < 5.0 Angstrom, "." denotes other aligned residues)
METGNKYIEKRAIDLSRERDPNFFDHPGIPVPECFWFMFKNNVRQDAGTCYSSWKMDMKVGPNWVHIKSDDNCNLSGDFPPGWIVLGKKRPGF*AFHLEPKTVELRVSMHCYGCAKKVQKHISKMDGVTSFEVDLESK-KVVVIGDITPYEVLASVS-KVMKFAELWVAPN----------------------*
........:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*    ........................................ .................. .............                      *
METGNKYIEKRAIDLSRERDPNFFDHPGIPVPECFWFMFKNNVRQDAGTCYSSWKMDMKVGPNWVHIKSDDNCNLSGDFPPGWIVLGKKRPGF*----MVQKIVIKVHMSSDKCRRKAMALAASTGGVVSVELAGDDRSKVVVVGDVDSIGLTNALRRKVDGSAELVEVSDASKKKEEEAKKKEEKEELVYYH*

#Total CPU time is  0.08 seconds

"""

import re
import subprocess
from pathlib import Path
from typing import Tuple


def calculate_tmscore(model:Path, native:Path) -> Tuple[int,float,float,float]:
    """
    Do a structural alignment of pdb1 onto pdb2 and calculate the TM-score,
    RMSD and the aligned length.
    """
    
    command = (f"USalign {model.resolve()} {native.resolve()} -TMscore 7 -ter 1").split()

    p = subprocess.run(command, capture_output=True)
    output_lines = p.stdout.decode().split("\n")

    aligned_length = int(re.search(r"Aligned length=\s+(\d+),", output_lines[13]
                                   ).group(1))
    rmsd = float(re.search(r"RMSD=\s+(\d+\.\d+),",output_lines[13]
                           ).group(1))
    tmscore_model = float(re.search(r"^TM-score=\s+(\d+\.\d+)", output_lines[14]
                               ).group(1))
    tmscore_native = float(re.search(r"^TM-score=\s+(\d+\.\d+)", output_lines[15]
                               ).group(1))
    
    return aligned_length, rmsd, tmscore_model, tmscore_native
