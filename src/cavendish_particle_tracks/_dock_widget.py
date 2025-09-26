# cavendish_particle_tracks/_dock_widget.py
from ._main_widget import get_singleton
from napari import current_viewer

def make_qwidget(viewer=None):
    """
    Factory function for Napari plugin system.

    - If Napari passes a viewer (as it does when using widgets:), use it.
    - If called from a menu command without a viewer, fall back to napari.current_viewer().
    """
    if viewer is None:
        viewer = current_viewer()
        if viewer is None:
            raise RuntimeError("No Napari viewer instance available to create ParticleTracksWidget")
    return get_singleton(viewer)
