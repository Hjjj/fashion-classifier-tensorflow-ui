import unittest
from logic import MNISTLogic

class TestMNISTLogic(unittest.TestCase):
    def setUp(self):
        self.logic = MNISTLogic()

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