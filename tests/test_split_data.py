from unittest import TestCase


class UnitTest(TestCase):
    def test__split_classes(self):
        from src.stages.split_data import _split_classes
        class_counts = {
            "A" : 15,
            "B" : 6,
            "C" : 18,
            "D" : 16,
            "E" : 32,
            "F" : 30,
            "G" : 10,
            "H" : 15,
            "1" : 15,
            "2" : 15,
            "3" : 15,
            "4" : 15,
            "5" : 15,
            "6" : 15,
        }
        train_fracture, val_fracture, test_fracture = 0.7, 0.2, 0.1
        class_names = []
        for key in class_counts.keys():
            class_names += [key] * class_counts[key]

        train_classes, val_classes, test_classes = _split_classes(
            class_names,
            train_fracture, val_fracture, test_fracture,
        )
        train_count = sum([ class_counts[key] for key in train_classes ])
        val_count = sum([ class_counts[key] for key in val_classes ])
        test_count = sum([ class_counts[key] for key in test_classes ])
        total_count = sum(class_counts.values())
        self.assertEqual(total_count, train_count + val_count + test_count)
        self.assertAlmostEqual(train_fracture, train_count / total_count, delta=0.05)
        self.assertAlmostEqual(val_fracture, val_count / total_count, delta=0.05)
        self.assertAlmostEqual(test_fracture, test_count / total_count, delta=0.05)
