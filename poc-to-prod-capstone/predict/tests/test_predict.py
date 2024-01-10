import unittest
from unittest.mock import MagicMock, patch
from text_prediction_model import TextPredictionModel

class TestTextPredictionModel(unittest.TestCase):
    def setUp(self):
        # Mock data for testing
        self.model_mock = MagicMock()
        self.params_mock = {"param1": "value1", "param2": "value2"}
        self.labels_to_index_mock = {"label1": 0, "label2": 1}
        self.labels_index_inv_mock = {0: "label1", 1: "label2"}

    def test_init(self):
        # Test initializing the TextPredictionModel instance
        model = TextPredictionModel(self.model_mock, self.params_mock, self.labels_to_index_mock)

        # Assert that the attributes are set correctly
        self.assertEqual(model.model, self.model_mock)
        self.assertEqual(model.params, self.params_mock)
        self.assertEqual(model.labels_to_index, self.labels_to_index_mock)
        self.assertEqual(model.labels_index_inv, self.labels_index_inv_mock)

    @patch("text_prediction_model.load_model")
    @patch("text_prediction_model.json.load")
    def test_from_artefacts(self, json_load_mock, load_model_mock):
        # Mock the return values of load_model and json.load
        load_model_mock.return_value = self.model_mock
        json_load_mock.side_effect = [
            self.params_mock,
            self.labels_to_index_mock
        ]

        # Call the class method from_artefacts
        model = TextPredictionModel.from_artefacts("fake_artefacts_path")

        # Assert that the attributes are set correctly
        self.assertEqual(model.model, self.model_mock)
        self.assertEqual(model.params, self.params_mock)
        self.assertEqual(model.labels_to_index, self.labels_to_index_mock)
        self.assertEqual(model.labels_index_inv, self.labels_index_inv_mock)

    @patch("text_prediction_model.embed")
    def test_predict(self, embed_mock):
        # Mock the return value of the embed function
        embed_mock.return_value = [[1.0, 2.0], [3.0, 4.0]]

        # Mock data for testing
        text_list = ["text1", "text2"]
        top_k = 2

        # Create an instance of TextPredictionModel for testing
        model = TextPredictionModel(self.model_mock, self.params_mock, self.labels_to_index_mock)

        # Call the predict method
        predictions = model.predict(text_list, top_k)

        # Assert that the predictions are as expected
        self.assertEqual(predictions, [['label2', 'label1'], ['label2', 'label1']])

if __name__ == "__main__":
    unittest.main()
