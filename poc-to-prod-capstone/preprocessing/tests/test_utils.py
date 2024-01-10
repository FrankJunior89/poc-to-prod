import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 180
        base._get_num_train_samples = MagicMock(return_value=180)
        # we assert that _get_num_train_batches will return 180/20 = 9
        self.assertEqual(base._get_num_train_batches(), 9)

    def test__get_num_test_samples(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 180
        base._get_num_samples = MagicMock(return_value=180)
        # we mock _get_num_train_samples to return the value 180
        base._get_num_train_samples = MagicMock(return_value=120)
        # we assert that _get_num_test_samples will return 180 - 120 = 60
        self.assertEqual(base._get_num_test_samples(), 60)

    def test__get_num_test_batches(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_test_samples to return the value 180
        base._get_num_test_samples = MagicMock(return_value=260)
        # we assert that _get_num_train_batches will return 180/20 = 13
        self.assertEqual(base._get_num_test_batches(), 13)



class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)

        self.assertEqual(dataset._get_num_samples(),2)




    if __name__ == '__main__':
        unittest.main()