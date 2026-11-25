import unittest

import pandas as pd

from model_pipeline import train_model


class TestTrainModel(unittest.TestCase):
    def test_model_can_predict(self):
        X_train = pd.DataFrame(
            {
                "Age": [20, 22, 40, 42],
                "Balance": [100, 150, 900, 1000],
            }
        )
        y_train = pd.Series([0, 0, 1, 1])

        model = train_model(X_train, y_train)
        predictions = model.predict(X_train)

        self.assertEqual(len(predictions), len(y_train))
        self.assertTrue(set(predictions).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
