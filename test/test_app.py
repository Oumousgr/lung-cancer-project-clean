# tests/test_app.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from app import load_and_prepare_data  # Assurez-vous d'importer les bonnes fonctions

class TestApp(unittest.TestCase):

    def test_data_loading(self):
        # Tester que la fonction de prétraitement retourne bien les données attendues
        X_train, X_test, y_train, y_test, X_test_final, y_test_final, scaler, column_names = load_and_prepare_data()
        self.assertEqual(len(X_train), len(y_train))  # Vérifie que les dimensions correspondent

if __name__ == "__main__":
    unittest.main()
