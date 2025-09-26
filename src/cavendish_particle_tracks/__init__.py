try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._main_widget import ParticleTracksWidget, get_singleton

__all__ = ("ParticleTracksWidget","get_singleton")
