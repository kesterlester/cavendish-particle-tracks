#!/usr/bin/env python3

# launch this file in debug mode in your desired IDE with breakpoints added where relevant
# launch_napari.py
#
# To turn on the opt-in interaction trace (for diagnosing bugs that only show up in a live
# mouse-driven session), set CPT_DEBUG_TRACE before launching, e.g.:
#     CPT_DEBUG_TRACE=1 ./launch_debug.py
# The log path is printed to the terminal on startup - see _main_widget.DEBUG_LOG_PATH.
from napari import Viewer, run

from cavendish_particle_tracks import get_singleton

viewer = Viewer()
plugin_docking_area = "bottom"

data_folder=None

# Developers may wish to uncomment one of the next two lines - the path depends on how deep this
# checkout sits relative to the sibling cavendish-particle-tracks-data/ repo. From the main repo
# root, one level up; from a git worktree living inside it (e.g. right-click-tree/), two.
# data_folder="../cavendish-particle-tracks-data/"          # main repo root
data_folder="../../cavendish-particle-tracks-data/"        # a worktree one level deeper

# Create the plugin:
plugin_widget = get_singleton(viewer, docking_area=plugin_docking_area, data_folder=data_folder )

# Add plugin to the viewer
dock_widget = viewer.window.add_dock_widget(
    plugin_widget, name="cavendish-particle-tracks", area=plugin_docking_area
)

run()
