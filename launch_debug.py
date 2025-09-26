#!/usr/bin/env python3

# launch this file in debug mode in your desired IDE with breakpoints added where relevant
# launch_napari.py
from napari import Viewer, run

from cavendish_particle_tracks import get_singleton

viewer = Viewer()
plugin_docking_area = "bottom"

data_folder=None

# Developers may wish to uncomment the next line
data_folder="../cavendish-particle-tracks-data/"

# Create the plugin:
plugin_widget = get_singleton(viewer, docking_area=plugin_docking_area, data_folder=data_folder )

# Add plugin to the viewer
dock_widget = viewer.window.add_dock_widget(
    plugin_widget, name="cavendish-particle-tracks", area=plugin_docking_area
)

run()
