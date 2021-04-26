import unittest.mock as mock
import os
import sys
import tempfile
import unittest
import pytest

from six.moves import StringIO

from smqtk_image_io.bbox import AxisAlignedBoundingBox
from smqtk_dataprovider.impls.data_element.file import DataFileElement
from smqtk_image_io.utils.image import (
    is_loadable_image,
    is_valid_element,
    crop_in_bounds,
)

from tests import TEST_DATA_DIR


class TestIsLoadableImage(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.good_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                      'Lenna.png'))
        cls.non_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                     'test_file.dat'))

    def test_non_data_element_raises_exception(self):
        # should throw:
        # AttributeError: 'bool' object has no attribute 'get_bytes'
        self.assertRaises(
            AttributeError,
            is_loadable_image, False
        )

    def test_unloadable_image_returns_false(self):
        assert is_loadable_image(self.non_image) is False

    def test_loadable_image_returns_true(self):
        assert is_loadable_image(self.good_image) is True


class TestIsValidElement(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.good_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                      'Lenna.png'))
        cls.non_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                     'test_file.dat'))

    def test_non_data_element(self):
        # Should check that input datum is a DataElement instance.
        # noinspection PyTypeChecker
        assert is_valid_element(False) is False

    def test_invalid_content_type(self):
        assert is_valid_element(self.good_image, valid_content_types=[]) \
               is False

    def test_valid_content_type(self):
        assert is_valid_element(self.good_image,
                                valid_content_types=['image/png']) is True

    def test_invalid_image_returns_false(self):
        assert is_valid_element(self.non_image, check_image=True) is False

class TestCropInBounds(object):
    """
    Test using the ``crop_in_bounds`` function.
    """

    def test_in_bounds_inside(self):
        """
        Test that ``in_bounds`` passes when crop inside given rectangle bounds.

            +--+
            |  |
            |##|  => (4, 6) image, (2,2) crop
            |##|
            |  |
            +--+

        """
        bb = AxisAlignedBoundingBox([1, 2], [3, 4])
        assert crop_in_bounds(bb, 4, 8)

    def test_in_bounds_inside_edges(self):
        """
        Test that a crop is "in bounds" when contacting the 4 edges of the
        given rectangular bounds.

            +--+
            |  |
            ## |  => (4, 6) image, (2,2) crop
            ## |
            |  |
            +--+

            +##+
            |##|
            |  |  => (4, 6) image, (2,2) crop
            |  |
            |  |
            +--+

            +--+
            |  |
            | ##  => (4, 6) image, (2,2) crop
            | ##
            |  |
            +--+

            +--+
            |  |
            |  |  => (4, 6) image, (2,2) crop
            |  |
            |##|
            +##+

        """
        # noinspection PyArgumentList
        bb = AxisAlignedBoundingBox([0, 2], [2, 4])
        assert crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, 0], [3, 2])
        assert crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([2, 2], [4, 4])
        assert crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, 4], [3, 6])
        assert crop_in_bounds(bb, 4, 6)

    def test_in_bounds_completely_outside(self):
        """
        Test that being completely outside the given bounds causes
        ``in_bounds`` to return False.
        """
        bb = AxisAlignedBoundingBox([100, 100], [102, 102])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([-100, -100], [-98, -98])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([-100, 100], [-98, 102])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([100, -100], [102, -98])
        assert not crop_in_bounds(bb, 4, 6)

    def test_in_bounds_crossing_edges(self):
        """
        Test that ``in_bounds`` returns False when crop crossed the 4 edges.

            +--+
            |  |
           ### |  => (4, 6) image, (3,2) crop
           ### |
            |  |
            +--+

            +--+
            |  |
            | ###  => (4, 6) image, (3,2) crop
            | ###
            |  |
            +--+

             ##
            +##+
            |##|
            |  |  => (4, 6) image, (2,3) crop
            |  |
            |  |
            +--+

            +--+
            |  |
            |  |  => (4, 6) image, (2,3) crop
            |  |
            |##|
            +##+
             ##

        """
        bb = AxisAlignedBoundingBox([-1, 2], [2, 4])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([2, 2], [5, 4])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, -1], [3, 2])
        assert not crop_in_bounds(bb, 4, 6)

        bb = AxisAlignedBoundingBox([1, 4], [3, 7])
        assert not crop_in_bounds(bb, 4, 6)

    def test_in_bounds_zero_crop_area(self):
        """
        Test that crop is not ``in_bounds`` when it has zero area (undefined).
        """
        # noinspection PyArgumentList
        bb = AxisAlignedBoundingBox([1, 2], [1, 2])
        assert not crop_in_bounds(bb, 4, 6)
