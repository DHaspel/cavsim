from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cavsim.base.components.numerical_component import NumericalComponent


class TestNumericalComponent(TestCase):

    def test___init__(self):
        nc = NumericalComponent()
        self.assertEqual(0.0, nc._delta_x)
        self.assertEqual(0.0, nc._delta_t)
        self.assertEqual({}, nc._fields)

    def test_fields(self):
        nc = NumericalComponent()
        self.assertEqual({}, nc.fields)
        nc._fields = None
        self.assertEqual(None, nc.fields)
        nc._fields = 123
        self.assertEqual(123, nc.fields)

    def test_field(self):
        nc = NumericalComponent()
        nc._fields = {'field1': (7,'abc'), 'field2': (9,'def')}
        # Invalid parameter tests
        with self.assertRaises(TypeError):
            nc.field(123)
        with self.assertRaises(KeyError):
            nc.field('NotAField')
        # Valid parameter tests
        self.assertEqual('abc', nc.field('field1'))
        self.assertEqual('def', nc.field('field2'))

    def test_field_slice(self):
        nc = NumericalComponent()
        f = nc.field_create('field', 3)
        nc.fields_resize(5)
        f[:,:] = np.reshape(np.arange(15), (3,5))[:,:]
        # Invalid parameter tests
        with self.assertRaises(TypeError):
            nc.field_slice(123, 0, 0)
        with self.assertRaises(TypeError):
            nc.field_slice('field', 'abc', 0)
        with self.assertRaises(TypeError):
            nc.field_slice('field', 0, 'xyz')
        # Out of bounds and other KeyErrors
        with self.assertRaises(KeyError):
            nc.field_slice('NotAField', 0, 0)
        with self.assertRaises(KeyError):
            nc.field_slice('field', 0, -2)
        with self.assertRaises(KeyError):
            nc.field_slice('field', 0, 2)
        with self.assertRaises(KeyError):
            nc.field_slice('field', -1, 0)
        with self.assertRaises(KeyError):
            nc.field_slice('field', 3, 0)
        # Valid parameter tests
        npt.assert_array_almost_equal([1,2,3], nc.field_slice('field', 0, 0))
        npt.assert_array_almost_equal([0,1,2], nc.field_slice('field', 0, -1))
        npt.assert_array_almost_equal([2,3,4], nc.field_slice('field', 0, 1))
        npt.assert_array_almost_equal([6,7,8], nc.field_slice('field', 1, 0))
        npt.assert_array_almost_equal([5,6,7], nc.field_slice('field', 1, -1))
        npt.assert_array_almost_equal([7,8,9], nc.field_slice('field', 1, 1))
        npt.assert_array_almost_equal([11,12,13], nc.field_slice('field', 2, 0))
        npt.assert_array_almost_equal([10,11,12], nc.field_slice('field', 2, -1))
        npt.assert_array_almost_equal([12,13,14], nc.field_slice('field', 2, 1))

    def test_field_create(self):
        # Invalid parameter tests
        nc = NumericalComponent()
        nc._fields = {'field1': (7,'abc'), 'field2': (9,'def')}
        with self.assertRaises(TypeError):
            nc.field_create(['field'], 7)
        with self.assertRaises(TypeError):
            nc.field_create('field', 'abc')
        with self.assertRaises(KeyError):
            nc.field_create('field2', 7)
        # Valid parameter tests
        nc = NumericalComponent()
        f = nc.field_create('field', 7)
        self.assertEqual(['field'], list(nc._fields.keys()))
        self.assertEqual(7, nc._fields['field'][0])
        npt.assert_array_equal(f, nc._fields['field'][1])

    def test_fields_resize(self):
        nc = NumericalComponent()
        # Invalid parameter tests
        with self.assertRaises(TypeError):
            nc.fields_resize('abc')
        # Valid parameter tests
        f1 = nc.field_create('field1', 1)
        f2 = nc.field_create('field2', 3)
        nc.fields_resize(5)
        self.assertEqual((1, 5), f1.shape)
        self.assertEqual((3, 5), f2.shape)

    def test_fields_move(self):
        # First test
        nc = NumericalComponent()
        f = nc.field_create('field', 3)
        nc.fields_resize(1)
        init = np.asarray([[1], [2], [3]])
        answer = np.asarray([[1], [1], [2]])
        f[:,:] = init[:,:]
        nc.fields_move()
        npt.assert_array_almost_equal(answer, f)
        # Second test
        nc = NumericalComponent()
        f = nc.field_create('field', 3)
        f2 = nc.field_create('field2', 4)
        nc.fields_resize(2)
        init = np.asarray([[1,9], [2,8], [3,7]])
        answer = np.asarray([[1,9], [1,9], [2,8]])
        f[:,:] = init[:,:]
        init2 = np.asarray([[1,6], [2,7], [3,8], [4,9]])
        answer2 = np.asarray([[1,6], [1,6], [2,7], [3,8]])
        f2[:,:] = init2[:,:]
        nc.fields_move()
        npt.assert_array_almost_equal(answer, f)
        npt.assert_array_almost_equal(answer2, f2)
