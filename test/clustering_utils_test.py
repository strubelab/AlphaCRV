"""
Unit tests for the functions in clustering_utils.py
"""

import unittest
import pandas as pd
from pathlib import Path
import sys
import os
import tempfile, shutil
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pickle
from unittest.mock import patch, MagicMock, Mock

sys.path.append(os.fspath(Path(__file__).parent.parent))

from alphacrv.clustering_utils import (get_scores, run_sequence_clustering,
                                           run_structure_clustering,
                                           get_topcluster_members,
                                           joint_cluster,
                                           merge_dict_values,
                                           align_all,
                                           medians_alignments,
                                           add_binder_fraction)

class GetScoresTest(unittest.TestCase):
    """
    Test the function get_scores
    """
    @patch('pickle.load', return_value={'iptm': 0.5})
    def test_valid_files(self, mock_pickle:MagicMock):
        """
        Test with valid files for the expected output of a pandas DataFrame with
        the iptm and iptm+ptm scores.

        Args:
            mock_pickle (MagicMock): Mock for pickle.load
        """
        models_dir = Path(__file__).parent / 'test_data/get_scores'
        expected_scores = pd.DataFrame({
            "complex": ["complex1", "complex2", "complex3", "complex4", "complex5"],
            "iptm": [0.5, 0.5, 0.5, 0.5, 0.5],
            "iptm+ptm": [0.2429, 0.3217, 0.3172, 0.3243, 0.2423]
        }).sort_values(by=['iptm', 'iptm+ptm'], ignore_index=True, ascending=False)
        actual_scores = get_scores(models_dir)
        self.assertTrue(actual_scores.equals(expected_scores))


@patch('subprocess.run')
@patch('pandas.read_table', return_value=pd.DataFrame({
    "rep": ["complex1", "complex2", "complex3"],
    "member": ["complex1", "complex2", "complex3"]
}))
class RunSequenceClusteringTest(unittest.TestCase):
    """
    Test the function run_sequence_clustering
    """
    
    def setUp(self):
        self.tempdir = Path(tempfile.gettempdir(),
            self.__class__.__name__.lower())
        self.tempdir.mkdir()
        self.destination = self.tempdir
        self.sequences = [SeqRecord(Seq('ACTG'), id='proteinA', description='proteinA'),
                          SeqRecord(Seq('ACTG'), id='proteinB', description='proteinB'),
                          SeqRecord(Seq('ACTG'), id='proteinC', description='proteinC')]
        self.results_prefix = 'clusterRes'
        self.temp_dir_name = 'tmp'
        self.min_seq_id = 0.3
        self.coverage = 0.8
        self.cov_mode = 2
        self.sensitivity = 7
        
        self.expected_clusters = pd.DataFrame({
            "rep": ["complex1", "complex2", "complex3"],
            "member": ["complex1", "complex2", "complex3"]
        })
    
    def test_valid_input(self, mock_read_table:MagicMock, mock_run:MagicMock):
        """
        Test with valid input for the expected output of a pandas DataFrame with
        the sequence clusters.

        Args:
            mock_read_table (MagicMock): Mock for pd.read_table
            mock_run (MagicMock): Mock for subprocess.run
        """
        out_seqcluster = self.destination / "seqclusters"
        sequences_file = out_seqcluster / "sequences.fasta"
        clusters = run_sequence_clustering(self.destination, self.sequences,
                                           self.results_prefix, self.temp_dir_name,
                                           self.min_seq_id, self.coverage,
                                           self.cov_mode, self.sensitivity)
        
        self.assertTrue(out_seqcluster.exists())
        self.assertTrue((out_seqcluster / "sequences.fasta").exists())
        
        with open(sequences_file, "r") as f:
            self.assertEqual(f.read(), ">proteinA\nACTG\n"
                                       ">proteinB\nACTG\n"
                                       ">proteinC\nACTG\n")
        
        command = (f"mmseqs easy-cluster {sequences_file.resolve()} {self.results_prefix}"
                   f" {self.temp_dir_name} --min-seq-id {self.min_seq_id} "
                   f"-c {self.coverage} --cov-mode {self.cov_mode} "
                   f"-s {self.sensitivity}").split()
        mock_run.assert_called_with(command, cwd=out_seqcluster,
                                    capture_output=True)
        
        self.assertTrue(clusters.equals(self.expected_clusters))
    
    def tearDown(self):
        if self.tempdir.exists():
            shutil.rmtree(self.tempdir)



@patch('subprocess.run')
@patch('pandas.read_table', return_value=pd.DataFrame({
    "rep": ["complex1", "complex2", "complex3"],
    "member": ["complex1", "complex2", "complex3"]
}))
class RunStructureClusteringTest(unittest.TestCase):
    """
    Test the function run_structure_clustering
    """
    
    def setUp(self):
        self.tempdir = Path(tempfile.gettempdir(),
            self.__class__.__name__.lower())
        self.tempdir.mkdir()
        self.destination = self.tempdir
        self.top_models_dir = Path('top_models')
        self.models_dir = Path('models')
        self.results_prefix = 'clusterRes'
        self.temp_dir_name = 'tmp'
        self.coverage = 0.8
        self.cov_mode = 2
        self.evalue = 0.01
        
        self.expected_clusters = pd.DataFrame({
            "rep": ["complex1", "complex2", "complex3"],
            "member": ["complex1", "complex2", "complex3"]
        })
    
    def test_valid_input1(self, mock_read_table:MagicMock, mock_run:MagicMock):
        """
        Test with valid input for the expected output of a pandas DataFrame with
        the structure clusters and the pdbs directory. Provide top_models_dir.

        Args:
            mock_read_table (MagicMock): Mock for pd.read_table
            mock_run (MagicMock): Mock for subprocess.run
        """
        out_strcluster = self.destination / "strclusters"
        clusters = run_structure_clustering(self.destination, self.top_models_dir,
                                            self.models_dir, self.results_prefix,
                                            self.temp_dir_name, self.coverage,
                                            self.cov_mode, self.evalue)
        
        self.assertTrue(out_strcluster.exists())
        
        command = (f"foldseek easy-cluster {self.top_models_dir.resolve()} {self.results_prefix}"
                   f" {self.temp_dir_name} "
                   f"-c {self.coverage} --cov-mode {self.cov_mode} "
                   f"-e {self.evalue}").split()
        mock_run.assert_called_with(command, cwd=out_strcluster,
                                    capture_output=True)
        
        self.assertEqual(clusters[1], self.top_models_dir)
        self.assertTrue(clusters[0].equals(self.expected_clusters))
    
    
    def mock_get_top_pdbs(models_dir:Path, destination:Path):
        """
        Replace the function get_top_pdbs for the test_valid_input2 test
        """
        return destination / 'all_pdbs'
    
    @patch('alphacrv.clustering_utils.get_top_pdbs', side_effect=mock_get_top_pdbs)
    def test_valid_input2(self, mock_get_top_pdbs:Mock, mock_read_table:MagicMock,
                          mock_run:MagicMock):
        """
        Test with valid input for the expected output of a pandas DataFrame with
        the structure clusters and the pdbs directory. Don't provide top_models_dir.

        Args:
            mock_read_table (MagicMock): Mock for pd.read_table
            mock_run (MagicMock): Mock for subprocess.run
        """
        out_strcluster = self.destination / "strclusters"
        clusters, pdbs_dir = run_structure_clustering(self.destination, None,
                                            self.models_dir, self.results_prefix,
                                            self.temp_dir_name, self.coverage,
                                            self.cov_mode, self.evalue)
        
        self.assertTrue(out_strcluster.exists())
        
        mock_get_top_pdbs.assert_called_with(self.models_dir, self.destination)
        
        command = (f"foldseek easy-cluster {pdbs_dir.resolve()} {self.results_prefix}"
                   f" {self.temp_dir_name} "
                   f"-c {self.coverage} --cov-mode {self.cov_mode} "
                   f"-e {self.evalue}").split()
        mock_run.assert_called_with(command, cwd=out_strcluster,
                                    capture_output=True)
        
        self.assertEqual(pdbs_dir, self.destination / 'all_pdbs')
        self.assertTrue(clusters.equals(self.expected_clusters))
    
    def tearDown(self):
        if self.tempdir.exists():
            shutil.rmtree(self.tempdir)
    

################ Tests for merging clusters ################

class GetTopClusterMembersTest(unittest.TestCase):
    """
    Class to test the get_topcluster_members function
    """
    
    def test_topstrclusters(self):
        
        strclusters_path = (Path(__file__).parent / \
                            'test_data/merging_clusters/strclusters_unmerged.csv')
        
        strclusters = pd.read_csv(strclusters_path)
        
        top_clusters_members = get_topcluster_members(strclusters)
        
        self.assertEqual(len(top_clusters_members), 73)
        
        test_set = {'A0A0P0X4R4', 'Q2R0K5', 'Q0J0N5', 'Q7XLA5', 'A0A0P0XGK8',
                    'A0A0P0WF19', 'A0A0P0YB14', 'A0A0P0XQK2'}
        self.assertEqual(top_clusters_members['A0A0P0XGK8.pdb_B'], test_set)
    
    def test_topseqclusters(self):
        
        seqclusters_path = (Path(__file__).parent / \
                            'test_data/merging_clusters/seqclusters_unmerged.csv')
        
        seqclusters = pd.read_csv(seqclusters_path)
        
        top_clusters_members = get_topcluster_members(seqclusters)
        
        self.assertEqual(len(top_clusters_members), 127)
        
        test_set = {'Q5VR67', 'Q9LG67'}
        self.assertEqual(top_clusters_members['Q5VR67'], test_set)
    


class JointClusterTest(unittest.TestCase):
    """
    Class to test the joint_cluster function
    """

    def test_non_overlapping_clusters1(self):
        """
        No overlap. Return the same four clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'C', 'D'}}
        strclusters_members = {'str1': {'E', 'F'}, 'str2': {'G', 'H'}}
        expected_result = {'seq1': {'A', 'B'},
                           'seq2': {'C', 'D'},
                           'str1': {'E', 'F'},
                           'str2': {'G', 'H'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
        
    def test_non_overlapping_clusters2(self):
        """
        No overlap. Return the same clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B', 'C'}, 'seq2': {'D', 'E', 'F'}}
        strclusters_members = {'str1': {'G', 'H', 'I'}, 'str2': {'J', 'K', 'L'}}
        expected_result = {'seq1': {'A', 'B', 'C'},
                           'seq2': {'D', 'E', 'F'},
                           'str1': {'G', 'H', 'I'},
                           'str2': {'J', 'K', 'L'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
    
    def test_overlapping_clusters1(self):
        """
        Return one big cluster.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'C', 'D'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'E'}}
        expected_result = {'str2': {'A', 'B', 'C', 'D', 'E'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
    
    def test_overlapping_clusters2(self):
        """
        Return two clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'D', 'E'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'F'}}
        expected_result = {'str1': {'A', 'B', 'C'},
                           'str2': {'D', 'E', 'F'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
        
        
    def test_overlapping_clusters3(self):
        """
        Return three clusters, with one sequence cluster intact.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'D', 'E'}, 'seq3': {'F', 'G'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'H'}}
        expected_result = {'str1': {'A', 'B', 'C'},
                           'str2': {'D', 'E', 'H'},
                           'seq3': {'F', 'G'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)


    def test_overlapping_clusters4(self):
        """
        Return four clusters, with one sequence and one structure cluster intact.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'D', 'E'}, 'seq3': {'F', 'G'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'H'}, 'str3': {'I', 'J'}}
        expected_result = {'str1': {'A', 'B', 'C'},
                           'str2': {'D', 'E', 'H'},
                           'str3': {'I', 'J'},
                           'seq3': {'F', 'G'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
    
    def test_overlapping_clusters5(self):
        """
        Return merged clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B', 'C', 'D'},
                               'seq2': {'E', 'F', 'G'},
                               'seq3': {'H', 'I'},
                               'seq4': {'J', 'K', 'L'}}
        strclusters_members = {'str1': {'B', 'C', 'E', 'L'},
                               'str2': {'D', 'H'},
                               'str3': {'I', 'M'},
                               'str4': {'N', 'O', 'P'}}
        
        expected_result = {'str1': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'},
                           'str4': {'N', 'O', 'P'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)


    def test_overlapping_clusters6(self):
        """
        Return merged clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B', 'C', 'D'},
                               'seq2': {'E', 'F', 'G'},
                               'seq3': {'H', 'I'},
                               'seq4': {'J', 'K', 'L'},
                               'seq5': {'Q', 'R', 'S', 'T'}}
        strclusters_members = {'str1': {'B', 'C', 'E', 'L'},
                               'str2': {'D', 'H'},
                               'str3': {'I', 'M'},
                               'str4': {'N', 'O', 'P'}}
        
        expected_result = {'str1': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'},
                           'str4': {'N', 'O', 'P'},
                           'seq5': {'Q', 'R', 'S', 'T'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('alphacrv.clustering_utils.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)


class MergeDictValues(unittest.TestCase):
    """
    Class to test the merge_dict_values function
    """
    
    def test_no_overlap(self):
        """
        Test a dictionary with no overlapping values
        """
        
        test_dict = {'A': {'a', 'b', 'c'},
                     'B': {'d', 'e', 'f'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 2)
        
        self.assertEqual(merged_dict['A'], {'a', 'b', 'c'})
        self.assertEqual(merged_dict['B'], {'d', 'e', 'f'})


    def test_overlap(self):
        """
        Test a dictionary with overlapping values
        """
        
        test_dict = {'A': {'a', 'b', 'c'},
                     'B': {'b', 'c', 'd'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 1)
        
        self.assertEqual(merged_dict['B'], {'a', 'b', 'c', 'd'})
    
    def test_overlap2(self):
        """
        Test a dictionary with overlapping values
        """
        
        test_dict = {'A': {'a', 'b', 'c'},
                     'B': {'b', 'c', 'd'},
                     'C': {'a', 'b', 'e', 'f'},
                     'D': {'g', 'h', 'i'},
                     'E': {'i', 'k', 'l'},
                     'F': {'m', 'n', 'o'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 3)
        self.assertEqual(merged_dict['C'], {'a', 'b', 'c', 'd', 'e', 'f'})
        self.assertEqual(merged_dict['E'], {'g', 'h', 'i', 'k', 'l'})
        self.assertEqual(merged_dict['F'], {'m', 'n', 'o'})
        
    
    def test_overlap3(self):
        """
        Test a dictionary with overlapping values
        """
        
        test_dict = {'str1': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L'},
                     'str2': {'A', 'B', 'C', 'D', 'H', 'I'},
                     'str3': {'H', 'I', 'M'},
                     'str4': {'N', 'O', 'P'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 2)
        self.assertEqual(merged_dict['str1'],
                         {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'})
        self.assertEqual(merged_dict['str4'], {'N', 'O', 'P'})


class AlignAllTest(unittest.TestCase):
    """
    Class to test the align_all function. The tests here need to run USalign,
    make sure the program is in the PATH.
    """
    
    def setUp(self) -> None:
        path = shutil.which("USalign")
        
        if path is None:
            raise Exception("USalign is not in the PATH. Please install it.")
        
        clusters_file = (Path(__file__).parent / \
                         'test_data/merging_clusters/merged_clusters.csv')
        
        self.clusters = pd.read_csv(clusters_file)
        
    
    def test_single_cluster(self):
        """
        Test that a single cluster is aligned all vs all.
        """
        clusters = self.clusters[self.clusters.merged_rep=='A0A0P0XE00.pdb_B']
        pdbs_dir = Path(__file__).parent / 'test_data/merging_clusters/trimmed_pdbs'
        result = align_all(clusters, pdbs_dir, cpus=4)

        expected_path = (Path(__file__).parent / 'test_data/merging_clusters/' \
                                                'alignments_cluster1.pkl')
        with open(expected_path, 'rb') as f:
            expected_result = pickle.load(f)

        self.assertTrue(expected_result.equals(result))


    def test_single_cluster2(self):
        """
        Test that a single cluster is aligned all vs all
        """
        clusters = self.clusters[self.clusters.merged_rep=='Q65XD1.pdb_B']
        pdbs_dir = Path(__file__).parent / 'test_data/merging_clusters/trimmed_pdbs'
        result = align_all(clusters, pdbs_dir, cpus=4)
        
        expected_path = (Path(__file__).parent / 'test_data/merging_clusters/' \
                                                'alignments_cluster2.pkl')
        with open(expected_path, 'rb') as f:
            expected_result = pickle.load(f)

        self.assertTrue(expected_result.equals(result))


    def test_two_clusters1(self):
        """
        Test that two clusters are aligned all vs all
        """
        select_clusters = ['A0A0P0XE00.pdb_B', 'Q65XD1.pdb_B']
        clusters = self.clusters[self.clusters.merged_rep.isin(select_clusters)]
        pdbs_dir = Path(__file__).parent / 'test_data/merging_clusters/trimmed_pdbs'
        result = align_all(clusters, pdbs_dir, cpus=4)

        expected_path = (Path(__file__).parent / 'test_data/merging_clusters/' \
                                                'alignments_cluster1.pkl')
        with open(expected_path, 'rb') as f:
            expected_result1 = pickle.load(f)
        
        expected_path = (Path(__file__).parent / 'test_data/merging_clusters/' \
                                                'alignments_cluster2.pkl')
        with open(expected_path, 'rb') as f:
            expected_result2 = pickle.load(f)
        
        expected_result = pd.concat([expected_result1, expected_result2]).reset_index(drop=True)
        self.assertTrue(expected_result.equals(result))
        



class MedianAlignmentsTest(unittest.TestCase):
    """
    Class to test the align_all function
    """
    
    def setUp(self) -> None:
        
        clusters_file = (Path(__file__).parent / \
                         'test_data/merging_clusters/merged_clusters.csv')
        
        self.clusters = pd.read_csv(clusters_file)

        alignments_path = (Path(__file__).parent / 'test_data/merging_clusters/' \
                                                'alignments_cluster1.pkl')
        with open(alignments_path, 'rb') as f:
            self.alignment_scores1 = pickle.load(f)
        
        alignments_path = (Path(__file__).parent / 'test_data/merging_clusters/' \
                                                'alignments_cluster2.pkl')
        with open(alignments_path, 'rb') as f:
            self.alignment_scores2 = pickle.load(f)
        
    
    def test_single_cluster_medians1(self):
        """
        Test that a single cluster is aligned all vs all
        """
        clusters = self.clusters[self.clusters.merged_rep=='A0A0P0XE00.pdb_B']
        median_scores = medians_alignments(self.alignment_scores1, clusters)
        
        self.assertEqual(median_scores.shape[0], clusters.shape[0])
        self.assertEqual(set(clusters.member), set(median_scores.member))

        # Test two medians
        prot1 = median_scores.iloc[0].member
        median1 = pd.concat([
                    self.alignment_scores1[self.alignment_scores1.ref==prot1].tmscore_ref,
                    self.alignment_scores1[self.alignment_scores1.member==prot1].tmscore_m]).median()
        self.assertEqual(median_scores.iloc[0].tmscore, median1)

        prot2 = median_scores.iloc[1].member
        median2 = pd.concat([
                    self.alignment_scores1[self.alignment_scores1.ref==prot2].tmscore_ref,
                    self.alignment_scores1[self.alignment_scores1.member==prot2].tmscore_m]).median()
        self.assertEqual(median_scores.iloc[1].tmscore, median2)

    
    def test_single_cluster_medians2(self):
        """
        Test that a single cluster is aligned all vs all
        """
        clusters = self.clusters[self.clusters.merged_rep=='Q65XD1.pdb_B']
        median_scores = medians_alignments(self.alignment_scores2, clusters)
        
        self.assertEqual(median_scores.shape[0], clusters.shape[0])
        self.assertEqual(set(clusters.member), set(median_scores.member))

        # Test two medians
        prot1 = median_scores.iloc[0].member
        median1 = pd.concat([
                    self.alignment_scores2[self.alignment_scores2.ref==prot1].tmscore_ref,
                    self.alignment_scores2[self.alignment_scores2.member==prot1].tmscore_m]).median()
        self.assertEqual(median_scores.iloc[0].tmscore, median1)

        prot2 = median_scores.iloc[1].member
        median2 = pd.concat([
                    self.alignment_scores2[self.alignment_scores2.ref==prot2].tmscore_ref,
                    self.alignment_scores2[self.alignment_scores2.member==prot2].tmscore_m]).median()
        self.assertEqual(median_scores.iloc[1].tmscore, median2)
    

    def test_two_cluster_medians1(self):
        """
        Test that a single cluster is aligned all vs all
        """
        select_clusters = ['A0A0P0XE00.pdb_B', 'Q65XD1.pdb_B']
        clusters = self.clusters[self.clusters.merged_rep.isin(select_clusters)]
        alignment_scores = pd.concat([self.alignment_scores1, self.alignment_scores2]
                                    ).reset_index(drop=True)
        median_scores = medians_alignments(alignment_scores, clusters)
        
        self.assertEqual(median_scores.shape[0], clusters.shape[0])
        self.assertEqual(set(clusters.member), set(median_scores.member))
        
        # Test two medians
        prot1 = median_scores.iloc[0].member
        median1 = pd.concat([
                    alignment_scores[alignment_scores.ref==prot1].tmscore_ref,
                    alignment_scores[alignment_scores.member==prot1].tmscore_m]).median()
        self.assertEqual(median_scores.iloc[0].tmscore, median1)

        prot2 = median_scores.iloc[-1].member
        median2 = pd.concat([
                    alignment_scores[alignment_scores.ref==prot2].tmscore_ref,
                    alignment_scores[alignment_scores.member==prot2].tmscore_m]).median()
        self.assertEqual(median_scores.iloc[-1].tmscore, median2)


class MockPDBModel:
    """
    Mock PDBModel class to return different sequence lengths
    """
    def __init__(self, seq_length):
        if isinstance(seq_length, int):
            self.seq = 'A' * seq_length
        
    def takeChains(self, index):
        if index[0] == 0:
            return MockPDBModel(50)
        elif index[0] == 1:
            return MockPDBModel(100)

    def sequence(self):
        return self.seq

@patch('biskit.PDBModel', new=MockPDBModel)
class AddBinderFractionTest(unittest.TestCase):
    """
    Test the add_binder_fraction function
    """

    def test_single_member1(self):
        median_scores = pd.DataFrame({
            'member': ['member1'],
            'aligned_length': [150]
        })
        pdbs_dir = Path('.')
        result = add_binder_fraction(median_scores, pdbs_dir)
        expected_result = pd.DataFrame({
            'member': ['member1'],
            'aligned_length': [150],
            'fraction_binder': [1.0]
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_single_member2(self):
        median_scores = pd.DataFrame({
            'member': ['member1'],
            'aligned_length': [60]
        })
        pdbs_dir = Path('.')
        result = add_binder_fraction(median_scores, pdbs_dir)
        expected_result = pd.DataFrame({
            'member': ['member1'],
            'aligned_length': [60],
            'fraction_binder': [0.1]
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_multiple_members1(self):
        median_scores = pd.DataFrame({
            'member': ['member1', 'member2'],
            'aligned_length': [70, 100]
        })
        pdbs_dir = Path('.')
        result = add_binder_fraction(median_scores, pdbs_dir)
        expected_result = pd.DataFrame({
            'member': ['member1', 'member2'],
            'aligned_length': [70, 100],
            'fraction_binder': [0.2, 0.5]
        })
        pd.testing.assert_frame_equal(result, expected_result)
    
    def test_multiple_members2(self):
        
        medians_file = (Path(__file__).parent / \
                         'test_data/merging_clusters/median_scores.csv')
        median_scores = pd.read_csv(medians_file)
        
        pdbs_dir = Path('.')
        result = add_binder_fraction(median_scores, pdbs_dir)
        
        self.assertEqual(result.iloc[0].fraction_binder, 0.5)



if __name__ == '__main__':
    unittest.main()
