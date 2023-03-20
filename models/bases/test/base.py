import unittest
from unittest.mock import patch
from ..base_class import Base

class TestBase(unittest.TestCase):

    def setUp(self):
        self.base_model = Base()

    def test_init(self):
        self.assertIsInstance(self.base_model, Base)
        self.assertEqual(self.base_model.name, 'base_model')

    def test_str(self):
        self.assertEqual(str(self.base_model), 'base_model')

    def test_set_name(self):
        class Derived(Base):
            def _set_name(self):
                return 'derived_model'
        derived_model = Derived()
        self.assertEqual(derived_model.name, 'derived_model')

    def test_init_print(self):
        with patch('builtins.print') as mock_print:
            base_model = Base()
            mock_print.assert_called_with('Created base_model')

if __name__ == '__main__':
    p = patch.multiple(Base, __abstractmethods__=set())
    p.start()
    unittest.main()
    p.stop()