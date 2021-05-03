from distutils.version import LooseVersion
import os
import pickle
import re
import unittest
import warnings

import numpy
import pytest
import unittest.mock as mock

from smqtk_image_io.impls.image_reader.gdal_io import (
    osgeo,
    get_gdal_driver_supported_mimetypes,
    load_dataset_tempfile,
    load_dataset_vsimem,
    GdalImageReader
)
from smqtk_image_io.impls.image_reader import gdal_io
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from smqtk_dataprovider.impls.data_element.file import DataFileElement
from smqtk_dataprovider.impls.data_element.memory import DataMemoryElement
from smqtk_core.configuration import configuration_test_helper

from tests import TEST_DATA_DIR


@pytest.mark.skipif(osgeo is None,
                    reason="osgeo module not importable.")
class TestGdalHelperFunctions (unittest.TestCase):
    """
    Handles gdal function testing
    """

    # keeping this variable split between the two classes
    # Couldn't see a good way to make the GH_ variables global while keeping
    # unit tests happy. Favoring specificity over redundancy

    gh_image_fp: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.gh_image_fp = os.path.join(TEST_DATA_DIR, "grace_hopper.png")

    def test_gdal_supported_drivers(self) -> None:
        """
        Test that GDAL driver mimetype set return is non-zero.
        """
        ret = get_gdal_driver_supported_mimetypes()
        assert isinstance(ret, set)
        assert len(ret) > 0

    def test_gdal_supported_drivers_caching(self) -> None:
        """
        Test that GDAL driver mimetype getter performs caching.
        """
        # If the expected cache attr exists already on the function, remove it
        if hasattr(get_gdal_driver_supported_mimetypes, 'cache'):
            del get_gdal_driver_supported_mimetypes.cache  # type: ignore
        assert not hasattr(get_gdal_driver_supported_mimetypes, 'cache')

        ret1 = get_gdal_driver_supported_mimetypes()

        # A second call to the function should return the same thing but not
        # call anything from GDAL.
        with mock.patch('smqtk_image_io.impls.image_reader.gdal_io.gdal') as m_gdal:
            ret2 = get_gdal_driver_supported_mimetypes()
            assert ret2 == ret1
            m_gdal.GetDriverCount.assert_not_called()
            m_gdal.GetDriver.assert_not_called()

    def test_load_dataset_tempfile(self) -> None:
        """
        Test DataElement temporary file based context loader.
        """
        # Creating separate element from global so we can mock it up.
        e = DataFileElement(self.gh_image_fp, readonly=True)
        e.write_temp = mock.MagicMock(wraps=e.write_temp)  # type: ignore
        e.clean_temp = mock.MagicMock(wraps=e.clean_temp)  # type: ignore
        e.get_bytes = mock.MagicMock(wraps=e.get_bytes)    # type: ignore

        # Using explicit patcher start/stop in order to avoid using ``patch``
        # as a decorator because ``osgeo`` might not be defined when
        # decorating the method.
        patcher_gdal_open = mock.patch(
                        'smqtk_image_io.impls.image_reader.gdal_io.gdal.Open',
                        wraps=osgeo.gdal.Open)
        self.addCleanup(patcher_gdal_open.stop)

        m_gdal_open = patcher_gdal_open.start()

        with load_dataset_tempfile(e) as gdal_ds:
            # noinspection PyUnresolvedReferences
            e.write_temp.assert_called_once_with()
            # noinspection PyUnresolvedReferences
            e.get_bytes.assert_not_called()

            m_gdal_open.assert_called_once()

            assert gdal_ds.RasterCount == 3
            assert gdal_ds.RasterXSize == 512
            assert gdal_ds.RasterYSize == 600

        # noinspection PyUnresolvedReferences
        e.clean_temp.assert_called_once_with()
        assert len(e._temp_filepath_stack) == 0

    def test_load_dataset_vsimem(self) -> None:
        """
        Test that VSIMEM loading context
        """
        if int(LooseVersion(osgeo.__version__).version[0]) < 2:
            pytest.skip("Skipping VSIMEM test because GDAL version < 2")

        # Creating separate element from global so we can mock it up.
        e = DataFileElement(self.gh_image_fp, readonly=True)
        e.write_temp = mock.MagicMock(wraps=e.write_temp)  # type: ignore
        e.clean_temp = mock.MagicMock(wraps=e.clean_temp)  # type: ignore
        e.get_bytes = mock.MagicMock(wraps=e.get_bytes)    # type: ignore

        vsimem_path_re = re.compile(r'^/vsimem/\w+$')

        # Using explicit patcher start/stop in order to avoid using ``patch``
        # as a *decorator* because ``osgeo`` might not be defined when
        # decorating the method.
        patcher_gdal_open = mock.patch(
            'smqtk_image_io.impls.image_reader.gdal_io.gdal.Open',
            wraps=osgeo.gdal.Open,
        )
        self.addCleanup(patcher_gdal_open.stop)
        patcher_gdal_unlink = mock.patch(
            'smqtk_image_io.impls.image_reader.gdal_io.gdal.Unlink',
            wraps=osgeo.gdal.Unlink,
        )
        self.addCleanup(patcher_gdal_unlink.stop)

        m_gdal_open = patcher_gdal_open.start()
        m_gdal_unlink = patcher_gdal_unlink.start()

        with load_dataset_vsimem(e) as gdal_ds:
            # noinspection PyUnresolvedReferences
            e.write_temp.assert_not_called()
            # noinspection PyUnresolvedReferences
            e.get_bytes.assert_called_once_with()

            m_gdal_open.assert_called_once()
            ds_path = gdal_ds.GetDescription()
            assert vsimem_path_re.match(ds_path)

            assert gdal_ds.RasterCount == 3
            assert gdal_ds.RasterXSize == 512
            assert gdal_ds.RasterYSize == 600

        m_gdal_unlink.assert_called_once_with(ds_path)
        # noinspection PyUnresolvedReferences
        e.clean_temp.assert_not_called()
        assert len(e._temp_filepath_stack) == 0

    def test_possible_gdal_gci_values_caching(self) -> None:
        """
        Test that ``get_possible_gdal_gci_values`` caches correctly.
        :return:
        """
        # Clear expected cache attribute
        if hasattr(gdal_io.get_possible_gdal_gci_values, 'cache'):
            del gdal_io.get_possible_gdal_gci_values.cache  # type: ignore

        assert not hasattr(gdal_io.get_possible_gdal_gci_values, 'cache')
        v1 = gdal_io.get_possible_gdal_gci_values()
        assert hasattr(gdal_io.get_possible_gdal_gci_values, 'cache')
        assert gdal_io.get_possible_gdal_gci_values() == v1

        # Place a dummy expected value into cache to make sure that its what is
        # returned
        gdal_io.get_possible_gdal_gci_values.cache = 'expected value'  # type: ignore
        assert gdal_io.get_possible_gdal_gci_values() == 'expected value'

        # Clear expected cache attribute for future calls.
        if hasattr(gdal_io.get_possible_gdal_gci_values, 'cache'):
            del gdal_io.get_possible_gdal_gci_values.cache  # type: ignore

    def test_get_gdal_gci_abbreviation_map_caching(self) -> None:
        """
        Test that ``get_gdal_gci_abbreviation_map`` returns a dictionary
        """
        # Clear expected cache attribute if there
        if hasattr(gdal_io.get_gdal_gci_abbreviation_map, 'map_cache'):
            del gdal_io.get_gdal_gci_abbreviation_map.map_cache  # type: ignore

        assert not hasattr(gdal_io.get_gdal_gci_abbreviation_map, 'map_cache')
        v1 = gdal_io.get_gdal_gci_abbreviation_map()
        assert hasattr(gdal_io.get_gdal_gci_abbreviation_map, 'map_cache')
        assert gdal_io.get_gdal_gci_abbreviation_map() == v1

        # Place dummy expected value into cache to make sure that its what is
        # returned
        gdal_io.get_gdal_gci_abbreviation_map.map_cache = 'expected value'  # type: ignore
        assert gdal_io.get_gdal_gci_abbreviation_map() == 'expected value'

        # Clear expected cache attribute if there for future calls.
        if hasattr(gdal_io.get_gdal_gci_abbreviation_map, 'map_cache'):
            del gdal_io.get_gdal_gci_abbreviation_map.map_cache  # type: ignore


def test_GdalImageReader_usable() -> None:
    """
    Test that GdalImageReader class reports as usable when GDAL is importable.
    """
    # Mock module value of ``osgeo`` to something not None to simulate
    # something having been imported.
    with mock.patch.dict('smqtk_image_io.impls.image_reader.gdal_io.__dict__',
                         {'osgeo': object()}):
        assert GdalImageReader.is_usable() is True


def test_GdalImageReader_not_usable_missing_osgeo() -> None:
    """
    Test that class reports as not usable when GDAL is not importable (set
    to None in module).
    """
    # Catch expected warnings to not pollute output.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        # Mock module value of ``osgeo`` to None.
        with mock.patch.dict('smqtk_image_io.impls.image_reader.gdal_io.__dict__',
                             {'osgeo': None}):
            assert GdalImageReader.is_usable() is False


@pytest.mark.skipif(not GdalImageReader.is_usable(),
                    reason="GdalImageReader implementation is not usable in "
                           "the current environment.")
class TestGdalImageReader (unittest.TestCase):
    """
    Tests for ``GdalImageReader`` class.

    If this was not skipped, ``osgeo`` must have been importable.
    """
    gh_image_fp: str
    gh_file_element: DataFileElement
    gh_cropped_image_fp: str
    gh_cropped_file_element: DataFileElement
    gh_cropped_bbox: AxisAlignedBoundingBox

    @classmethod
    def setUpClass(cls) -> None:
        # Initialize test image paths/elements/associated crop boxes.

        cls.gh_image_fp = os.path.join(TEST_DATA_DIR, "grace_hopper.png")
        cls.gh_file_element = DataFileElement(cls.gh_image_fp, readonly=True)
        assert cls.gh_file_element.content_type() == 'image/png'

        cls.gh_cropped_image_fp = \
            os.path.join(TEST_DATA_DIR, 'grace_hopper.100x100+100+100.png')
        cls.gh_cropped_file_element = \
            DataFileElement(cls.gh_cropped_image_fp, readonly=True)
        assert cls.gh_cropped_file_element.content_type() == 'image/png'
        cls.gh_cropped_bbox = AxisAlignedBoundingBox([100, 100], [200, 200])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.gh_file_element.clean_temp()
        cls.gh_cropped_file_element.clean_temp()

    def test_init_default(self) -> None:
        """
        Test that construction with default parameters works.
        """
        GdalImageReader()

    def test_init_bad_load_method(self) -> None:
        """
        Test that passing a load_method string that is not one of the
        expected values raises a ValueError.
        """
        with pytest.raises(ValueError, match=r"Load method provided not a "
                                             r"valid value \(given 'not a "
                                             r"valid method'\)\. Must be one "
                                             r"of: "):
            GdalImageReader(load_method="not a valid method")

    @mock.patch("smqtk_image_io.impls.image_reader.gdal_io.osgeo")
    def test_init_vsimem_req_gdal_2_fail(self, m_osgeo: osgeo) -> None:
        """
        Test that we get a RuntimeError when using load_method='vsimem' and
        the current GDAL wrapper version is < 2.
        """
        # Mock a GDAL version less than 2.
        m_osgeo.__version__ = "1.11.0"
        with pytest.raises(RuntimeError,
                           match=r"Load method '{}' was specified, "
                                 r"but required GDAL version of 2 is not "
                                 r"satisfied \(imported version: {}\)\."
                                 .format(GdalImageReader.LOAD_METHOD_VSIMEM,
                                         "1.11.0")):
            GdalImageReader(load_method=GdalImageReader.LOAD_METHOD_VSIMEM)

    @mock.patch("smqtk_image_io.impls.image_reader.gdal_io.osgeo")
    def test_init_vsimem_req_gdal_2_pass(self, m_osgeo: osgeo) -> None:
        """
        Test that we do NOT get an error when using load_method='vsimem'
        and GDAL reports a version greater than 2.
        """
        # Mock a GDAL version greater than 2.
        m_osgeo.__version__ = "2.4.0"
        GdalImageReader(load_method=GdalImageReader.LOAD_METHOD_VSIMEM)

    def test_init_channel_order_lower_case(self) -> None:
        """
        Test that channel order if provided as string is cast to lower case.
        """
        assert GdalImageReader(channel_order='rGB')._channel_order == \
            "rgb"

    def test_init_channel_order_not_sequence(self) -> None:
        """
        Test that ``channel_order`` sequence check catches appropriate culprits.
        """
        expected_message = 'Channel order must be a sequence'
        with pytest.raises(ValueError, match=expected_message):
            # Sets should be invalid.
            # noinspection PyTypeChecker
            GdalImageReader(channel_order={1, 2, 3})  # type: ignore
        with pytest.raises(ValueError, match=expected_message):
            GdalImageReader(channel_order={1: 0, 2: 3})  # type: ignore

    def test_init_channel_order_empty_sequence(self) -> None:
        """
        Test that we get an error if an empty sequence is provided.
        """
        expected_message = "Invalid channel order, must request at least "\
                           "one band"
        with pytest.raises(ValueError, match=expected_message):
            GdalImageReader(channel_order='')
        with pytest.raises(ValueError, match=expected_message):
            GdalImageReader(channel_order=[])
        with pytest.raises(ValueError, match=expected_message):
            GdalImageReader(channel_order=())

    def test_init_channel_order_invalid_abbreviation_char(self) -> None:
        """
        Test that exception is raise upon construction with an unknown channel
        order abbreviation.
        """
        msg = 'Invalid abbreviation character'
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order='!')
        # Check that we catch it among other valid characters
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order='rg!')
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order='!gb')
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order='r!b')

    def test_init_channel_order_invalid_gci_integer(self) -> None:
        """
        Test that we catch integers that are not reported as supported constants
        by inspection of GDAL module values.
        """
        # We know (as of gdal 2.4.0) that 99 is not a valid GCI.
        msg = "Invalid GDAL band integer values given"
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order=[99])
        # Check that we catch it among other valid characters
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order=[3, 4, 99])
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order=[3, 99, 5])
        with pytest.raises(ValueError, match=msg):
            GdalImageReader(channel_order=[99, 4, 5])

    def test_configuration(self) -> None:
        """
        Test configuration using helper
        """
        expected_load_method = GdalImageReader.LOAD_METHOD_TEMPFILE
        expected_channel_order = [3, 5, 4]
        i = GdalImageReader(load_method=expected_load_method,
                            channel_order=expected_channel_order)
        for inst in configuration_test_helper(i):  # type: GdalImageReader
            assert inst._load_method == expected_load_method
            assert inst._channel_order == expected_channel_order

    def test_serialization(self) -> None:
        """
        Test that we can serialize and deserialize the algorithm, maintaining
        configuration responses.
        """
        expected_load_method = GdalImageReader.LOAD_METHOD_TEMPFILE
        expected_channel_order = [3, 5, 4]
        inst1 = GdalImageReader(load_method=expected_load_method,
                                channel_order=expected_channel_order)

        expected_config = {'load_method': expected_load_method,
                           'channel_order': expected_channel_order}
        assert inst1.get_config() == expected_config

        buff = pickle.dumps(inst1)
        #: :type: GdalImageReader
        inst2 = pickle.loads(buff)
        assert inst2.get_config() == expected_config

    @mock.patch('smqtk_image_io.impls.image_reader.gdal_io'
                '.get_gdal_driver_supported_mimetypes',
                wraps=get_gdal_driver_supported_mimetypes)
    def test_valid_content_types(self, m_ggdsm: mock.Mock) -> None:
        """
        Test that valid_content_types refers to the helper method and
        returns the same thing.

        Mocking (wrapping) `get_gdal_driver_supported_mimetypes` in order check
        that it is being called under the hood.
        """
        expected_content_types = get_gdal_driver_supported_mimetypes()

        actual_content_types = GdalImageReader().valid_content_types()

        m_ggdsm.assert_called_once_with()
        assert actual_content_types == expected_content_types

    def test_load_as_matrix_empty_data(self) -> None:
        """
        Test that we catch and do not load an empty data element.
        """
        empty_de = DataMemoryElement(readonly=True, content_type='image/png')
        assert empty_de.is_empty()
        msg = "GdalImageReader cannot load 0-sized data"
        with pytest.raises(ValueError, match=msg):
            GdalImageReader().load_as_matrix(empty_de)

    def test_load_as_matrix_tempfile(self) -> None:
        """
        Test that whole image is loaded successfully using tempfile loader.
        """
        wrapped_temp_loader = mock.MagicMock(wraps=load_dataset_tempfile)
        wrapped_vsimem_loader = mock.MagicMock(wraps=load_dataset_vsimem)

        with mock.patch.dict(GdalImageReader.LOAD_METHOD_CONTEXTMANAGERS, {
                    GdalImageReader.LOAD_METHOD_TEMPFILE: wrapped_temp_loader,
                    GdalImageReader.LOAD_METHOD_VSIMEM: wrapped_vsimem_loader
                }):
            # Using tempfile load method
            reader = GdalImageReader(
                load_method=GdalImageReader.LOAD_METHOD_TEMPFILE
            )
            mat = reader._load_as_matrix(self.gh_file_element)
            assert mat.shape == (600, 512, 3)

        wrapped_temp_loader.assert_called_once_with(self.gh_file_element)
        wrapped_vsimem_loader.assert_not_called()

    def test_load_as_matrix_vsimem(self) -> None:
        """
        Test that whole image is loaded successfully using vsimem loader.
        """
        if int(LooseVersion(osgeo.__version__).version[0]) < 2:
            pytest.skip("Skipping VSIMEM test because GDAL version < 2")

        wrapped_temp_loader = mock.MagicMock(wraps=load_dataset_tempfile)
        wrapped_vsimem_loader = mock.MagicMock(wraps=load_dataset_vsimem)

        with mock.patch.dict(GdalImageReader.LOAD_METHOD_CONTEXTMANAGERS, {
                    GdalImageReader.LOAD_METHOD_TEMPFILE: wrapped_temp_loader,
                    GdalImageReader.LOAD_METHOD_VSIMEM: wrapped_vsimem_loader
                }):
            # Using VSIMEM load method
            reader = GdalImageReader(
                load_method=GdalImageReader.LOAD_METHOD_VSIMEM
            )
            mat = reader._load_as_matrix(self.gh_file_element)
            assert mat.shape == (600, 512, 3)

        wrapped_temp_loader.assert_not_called()
        wrapped_vsimem_loader.assert_called_once_with(self.gh_file_element)

    def test_load_as_matrix_with_crop(self) -> None:
        """
        Test that the image is loaded with the correct crop region.

        We load two images: the original with a crop specified, and a
        pre-cropped image. The results of each load should be the same,
        indicating the correct region from the source image is extracted.
        """
        assert \
            self.gh_file_element.get_bytes() != self.gh_cropped_file_element.get_bytes()
        reader = GdalImageReader(
            load_method=GdalImageReader.LOAD_METHOD_TEMPFILE)
        cropped_actual = reader.load_as_matrix(self.gh_file_element,
                                               pixel_crop=self.gh_cropped_bbox)
        cropped_expected = reader.load_as_matrix(self.gh_cropped_file_element)
        # noinspection PyTypeChecker
        numpy.testing.assert_allclose(cropped_actual, cropped_expected)

    def test_load_as_matrix_with_crop_not_in_bounds(self) -> None:
        """
        Test that error is raised when crop bbox is not fully within the image
        bounds.
        """
        inst = GdalImageReader()

        # Nowhere close
        bb = AxisAlignedBoundingBox([5000, 6000], [7000, 8000])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside left side
        bb = AxisAlignedBoundingBox([-1, 1], [2, 2])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside top side
        bb = AxisAlignedBoundingBox([1, -1], [2, 2])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside right side
        bb = AxisAlignedBoundingBox([400, 400], [513, 600])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

        # Outside bottom side
        bb = AxisAlignedBoundingBox([400, 400], [512, 601])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(self.gh_file_element, pixel_crop=bb)

    def test_load_as_matrix_with_channel_order_invalid(self) -> None:
        """
        Test that we error when the requested channel order cannot be applied to
        the data given.
        """
        # We know that the self.gh_file_element image is an RGB format image.
        msg = "Data element did not provide channels required"
        with pytest.raises(RuntimeError, match=msg):
            GdalImageReader(channel_order=[gdal_io.gdal.GCI_CyanBand])\
                .load_as_matrix(self.gh_file_element)
        with pytest.raises(RuntimeError, match=msg):
            # Same as above, but with abbreviation
            GdalImageReader(channel_order='c')\
                .load_as_matrix(self.gh_file_element)

    def test_load_as_matrix_single_channel_from_multi(self) -> None:
        """
        Test that when requesting a single channel we get back an ndim=2 matrix
        instead of a matrix with a 1-valied dimension.
        """
        # Load image normally, assumed functional (see above)
        mat_rgb = GdalImageReader().load_as_matrix(self.gh_file_element)
        mat_b = GdalImageReader(channel_order='b')\
            .load_as_matrix(self.gh_file_element)
        assert mat_b is not None
        assert mat_b.shape == (600, 512)
        assert mat_b.ndim == 2
        numpy.testing.assert_allclose(mat_b, mat_rgb[:, :, 2])  # type: ignore

    def test_load_as_matrix_single_channel_from_multi_cropped(self) -> None:
        """
        Same as previous test but with crop region provided.
        """
        # Load image normally, assumed functional (see above)
        mat_rgb_cropped = GdalImageReader()\
            .load_as_matrix(self.gh_cropped_file_element)
        mat_b_cropped = GdalImageReader(channel_order='b')\
            .load_as_matrix(self.gh_file_element, pixel_crop=self.gh_cropped_bbox)
        assert mat_b_cropped is not None
        assert mat_b_cropped.ndim == 2
        assert mat_b_cropped.shape == (100, 100)
        numpy.testing.assert_allclose(mat_b_cropped,
                                      mat_rgb_cropped[:, :, 2])  # type: ignore

    def test_load_as_matrix_channel_subselection(self) -> None:
        """
        Test correctly loading image with only specific channels of what is
        available.
        """
        # Load image normally, assumed functional (see above)
        mat_rgb = GdalImageReader().load_as_matrix(self.gh_file_element)
        mat_gb = GdalImageReader(channel_order='gb')\
            .load_as_matrix(self.gh_file_element)
        numpy.testing.assert_allclose(mat_gb, mat_rgb[:, :, [1, 2]])  # type: ignore

    def test_load_as_matrix_channel_subselection_cropped(self) -> None:
        """
        Same as previous test but with crop region provided.
        """
        # Load image normally, assumed functional (see above)
        mat_rgb = GdalImageReader().load_as_matrix(self.gh_cropped_file_element)
        mat_gb = GdalImageReader(channel_order='gb') \
            .load_as_matrix(self.gh_file_element, pixel_crop=self.gh_cropped_bbox)
        assert mat_gb is not None
        assert mat_gb.ndim == 3
        assert mat_gb.shape == (100, 100, 2)
        numpy.testing.assert_allclose(mat_gb, mat_rgb[:, :, [1, 2]])  # type: ignore

    def test_load_as_matrix_channel_reorder(self) -> None:
        """
        Test correctly loading image with channels reordered as requested.
        """
        # Load image normally, assumed functional (see above)
        mat_rgb = GdalImageReader().load_as_matrix(self.gh_file_element)
        mat_brg = GdalImageReader(channel_order='brg')\
            .load_as_matrix(self.gh_file_element)
        numpy.testing.assert_allclose(mat_brg,
                                      mat_rgb[:, :, [2, 0, 1]])  # type: ignore

        # Duplicate bands? Because we can?
        mat_bgrgb = GdalImageReader(channel_order='bgrbg')\
            .load_as_matrix(self.gh_file_element)
        numpy.testing.assert_allclose(mat_bgrgb,
                                      mat_rgb[:, :, [2, 1, 0, 2, 1]])  # type: ignore

    def test_load_as_matrix_channel_reorder_cropped(self) -> None:
        """
        Same as previous test but with crop region provided.
        """
        # Load image normally, assumed functional (see above)
        mat_rgb = GdalImageReader().load_as_matrix(self.gh_cropped_file_element)
        mat_brg = GdalImageReader(channel_order='brg') \
            .load_as_matrix(self.gh_file_element, pixel_crop=self.gh_cropped_bbox)
        assert mat_brg is not None
        assert mat_brg.ndim == 3
        assert mat_brg.shape == (100, 100, 3)
        numpy.testing.assert_allclose(mat_brg,
                                      mat_rgb[:, :, [2, 0, 1]])  # type: ignore

        # Duplicate bands? Because we can?
        mat_bgrggb = GdalImageReader(channel_order='bgrggb') \
            .load_as_matrix(self.gh_file_element, pixel_crop=self.gh_cropped_bbox)
        assert mat_bgrggb is not None
        assert mat_bgrggb.ndim == 3
        assert mat_bgrggb.shape == (100, 100, 6)
        numpy.testing.assert_allclose(mat_bgrggb,
                                      mat_rgb[:, :, [2, 1, 0, 1, 1, 2]])  # type: ignore
