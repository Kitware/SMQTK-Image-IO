import pkg_resources

from .interfaces.image_reader import ImageReader
from .bbox import AxisAlignedBoundingBox

# It is known that this will fail if this package is not "installed" in the
# current environment. Additional support is pending defined use-case-driven
# requirements.
__version__ = pkg_resources.get_distribution(__name__).version
