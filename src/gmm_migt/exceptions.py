"""Exception types raised by GMM-MIGT."""

from __future__ import annotations


class GMMMIGTError(Exception):
    """Base class for all GMM-MIGT errors."""


class ImageLoadError(GMMMIGTError, ValueError):
    """An image file could not be read or decoded."""


class ImageShapeError(GMMMIGTError, ValueError):
    """Two images do not have the shape required for an MI comparison."""


class ConfigurationError(GMMMIGTError, ValueError):
    """An invalid parameter combination was supplied."""


class IntegrityError(GMMMIGTError):
    """A partition lost, duplicated or mis-copied an image."""
