import unittest
import os
from pathlib import Path
import sys
import numpy as np
import pickle

sys.path.append(os.fspath(Path(__file__).parent.parent))

from alphacrv.trimming_utils import get_border_indices, get_lowpae_indices

class GetBorderIndicesTest(unittest.TestCase):
    """
    Test the function get_border_indices
    """

    def test_all_below(self):
        mean_pae = np.array([6, 7, 8, 9, 10])
        pthresh = 15
        self.assertEqual(get_border_indices(mean_pae, pthresh), (0, 4))

    def test_none_below(self):
        mean_pae = np.array([16, 17, 18, 19, 20])
        pthresh = 10
        btleft, btright = get_border_indices(mean_pae, pthresh)
        self.assertTrue(np.isnan(btleft) and np.isnan(btright))

    def test_some_below(self):
        mean_pae = np.array([6, 7, 8, 9, 10])
        pthresh = 8
        self.assertEqual(get_border_indices(mean_pae, pthresh), (0, 2))


class GetLowPaeIndicesTest(unittest.TestCase):
    """
    Test the function get_lowpae_indices
    """

    def test_all_below(self):
        """
        All values in the pae matrix are below the first attempted threshold
        """
        # Generate a pae matrix of dimensions 30x30
        pae = np.zeros((30, 30))
        slength = 5
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 25, pthresh))

    def test_none_below(self):
        """
        No values in the pae matrix are below the first attempted threshold,
        but they are below the second
        """
        # Generate a pae matrix of dimensions 30x30
        pae = np.full((30, 30), 18)
        slength = 5
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 25, pthresh+5))

    def test_some_below(self):
        """
        No values in the pae matrix are below the first attempted threshold,
        less than 20 are below the second, and more than 20 are below the third
        """
        pae = np.full((30, 30), 23)
        slength = 5
        # Set some values in the lower left PAE quadrant to 18 and others to 30
        pae[slength:10, 0:slength] = 18
        pae[25:30, 0:slength] = 30
        # Set some values in the upper right PAE quadrant to 18 and others to 30
        pae[0:slength, slength:10] = 18
        pae[0:slength, 25:30] = 30
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 20, pthresh+10))
    
    def test_asymmetrical1(self):
        """
        The PAE matrix is asymmetrical, but values in lower left quadrant are
        below the threshold
        """
        pae = np.full((30, 30), 18)
        slength = 5
        # Set values in the lower left PAE quadrant to be below the threshold
        pae[slength:, :slength] = 10
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 25, pthresh))
    
    def test_asymmetrical2(self):
        """
        The PAE matrix is asymmetrical, but values in upper right quadrant are
        below the threshold
        """
        pae = np.full((30, 30), 18)
        slength = 5
        # Set values in the upper right PAE quadrant to be below the threshold
        pae[:slength, slength:] = 10
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 25, pthresh))
    
    def test_asymmetrical3(self):
        """
        Lower than 20 values in upper right quadrant are below the second
        threshold, but more than 20 values in the lower left quadrant are below
        """
        pae = np.full((30, 30), 23)
        slength = 5
        # Set values in the lower left PAE quadrant to 18
        pae[slength:, :slength] = 18
        # Set some values in the upper right PAE quadrant to 18
        pae[:slength, slength:slength+10] = 18
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 25, pthresh+5))
    
    def test_asymmetrical4(self):
        """
        Lower than 20 values in lower left quadrant are below the second
        threshold, but more than 20 values in the upper right quadrant are below
        """
        pae = np.full((30, 30), 23)
        slength = 5
        # Set values in the upper right PAE quadrant to 18
        pae[:slength, slength:] = 18
        # Set some values in the lower left PAE quadrant to 18
        pae[slength:slength+10, :slength] = 18
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 25, pthresh+5))
    
    def test_truedata(self):
        """
        Test the function with a true PAE matrix
        """
        pae_file = Path(__file__).parent / "test_data/trimming/pae_A0A0P0XXK2.pkl"
        with open(pae_file, 'rb') as f:
            pae = pickle.load(f)
        slength = 175
        pthresh = 15
        result = get_lowpae_indices(pae, slength, pthresh)
        self.assertEqual(result, (0, 32, pthresh))
        

if __name__ == '__main__':
    unittest.main()