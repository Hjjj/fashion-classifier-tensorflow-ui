import unittest
import tkinter as tk
from logic import MNISTLogic

class TestMNISTLogic(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()  # Initialize the Tkinter root window
        self.logic = MNISTLogic()

    def tearDown(self):
        self.root.destroy()  # Destroy the Tkinter root window after each test

    def test_initial_index(self):
        self.assertEqual(self.logic.index, 0)

    def test_get_next_image(self):
        image = self.logic.get_next_image()
        self.assertIsNotNone(image)
        self.assertEqual(self.logic.index, 1)

    def test_get_next_image_exhaustion(self):
        for _ in range(len(self.logic.x_train)):
            self.logic.get_next_image()
        image = self.logic.get_next_image()
        self.assertIsNone(image)

if __name__ == "__main__":
    unittest.main()