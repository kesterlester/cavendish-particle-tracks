
import numpy as np
import napari
import tempfile
import pandas as pd
from napari.layers.utils.stack_utils import stack_to_images

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidgetItem,
    QAction,
    QMenu,
    QInputDialog,
)
from qtpy.QtGui import QCursor, QMouseEvent
from qtpy.QtCore import Qt, QEvent, QTimer, QPoint

from .analysis import VIEW_NAMES, CalibrationData, FIDUCIAL_NAMES
from .napari_tools import (
    make_move_only,
    overwrite_layer,
)
from .tools import Accumulator

"""
Calibration points (locations of fiducials) come in two types:

(1) generic points (also called Workspace points)
(2) points in specific events

The former (Generic or workspace points) are not associated to any individual event, 
but serve as markers for roughtly where they might be, or serve to assist in the placement
of specific points -- i.e. points of the latter type. E.g. the former could be default 
locations for the latter before the latter are committed or tweaked.
"""

view_indices = (0, 1, 2)
GENERIC_CALIBRATION_LAYER_NAME = "Calibration (generic)"
PER_IMAGE_CALIBRATION_LAYER_NAME = "Calibration (per-image)"

class CalibrationManager:
    """
    This class stores, restores, and manages access to generic
    and specific calibration data.
    """

    num_generic_front_back_fid_pairs = 3

    def __init__(self, parent, viewer):

        self.parent = parent
        self.viewer = viewer
        self.calibration_data = CalibrationData()

        self.event_calibration_layer()  # Makes this layer and puts it earlier in the layer list than the next guys:
        # TODO: Try to avoid re-storing this redundant list of generic calibration layers .... should to live only in viewer?
        self._generic_calibration_layers = self._setup_calibration_layers()  # Returns a list of napari point layers.

        # Calibration layers start hidden - nothing to look at before an image is loaded. _load_data_from() in
        # _main_widget.py switches them on once loading has actually finished, to avoid a napari timing issue
        # where flipping visibility mid-image-load crashes the renderer.
        assert hasattr(self, "_generic_calibration_layers")
        self.set_calibration_layer_visibility_and_focus(False, False)

        self.mark_clean()

        # Lastly, setup callbacks:
        self._setup_callbacks()

    def _setup_calibration_layers(self):
        layers = self._default_generic_calibration_layers()

        # Overwrite data if layer already exists, otherwise make a note of new layers
        new_layers = []
        for layer in layers:
            if layer.name in self.viewer.layers:
                # Existing layer, so callbacks already exist too, so just overwrite old layer data:
                overwrite_layer(self.viewer.layers[layer.name], layer)
            else:
                # New layer!
                new_layers.append(layer)

        # Tell Napari about any new generic calibration layers:
        for new_layer in new_layers:
            self.viewer.add_layer(new_layer)

        return new_layers

    def mark_clean(self):
        import copy
        self.last_clean_state = copy.deepcopy(self.state())

    def dirty_things(self):
        calibrations_are_dirty = False

        #print("JJJJJJJJ", len(self.last_clean_state))
        #print("KKKKKKKK", len(self.state()))

        for a, A in zip(self.last_clean_state, self.state()):
            data, meta, _ = a
            DATA, META, _ = A
            if (data != DATA).any():
                calibrations_are_dirty = True
                break
            # TODO: Insert meta comparison to

        if calibrations_are_dirty:
            return [ "calibrations" ]
        else:
            return []

    def state(self):
        ans = tuple(l.as_layer_data_tuple() for l in self.generic_calibration_layers()) + \
                            ( self.event_calibration_layer().as_layer_data_tuple(), ) # Don't forget that comma!
        return ans

    def event_calibration_layer(self) -> napari.layers.Points:
        # TODO: This could break if the user first created a layer with exactly the right name before we construct.
        # Maybe should not reference by NAME but keep a reference.
        if PER_IMAGE_CALIBRATION_LAYER_NAME in self.parent.viewer.layers:
            # Layer already exists, so just return it:
            return self.parent.viewer.layers[PER_IMAGE_CALIBRATION_LAYER_NAME]

        # Layer does not already exist, so construct and return it:
        layer = self.parent.viewer.add_points(name=PER_IMAGE_CALIBRATION_LAYER_NAME, ndim=4, visible=False)
        layer.properties = {"labels": np.array([], dtype=object), }
        print(f"MOO SEE {layer.properties}")
        layer.text = {
            'string': 'labels',
            # This is a key in properties. Somehow it causes the error "Applying the encoding failed. Using the fallback value instead."
            'color': 'white',
            'size': 12,  # text size
            'anchor': 'center',
            'translation': np.array([0, 0, -150, 0]),  # move text 150 (data) pixels up.  4D since 4D
        }
        layer.events.data.connect(self._sync_calibration_data_from_event_layer)
        return layer

    def _sync_calibration_data_from_event_layer(self, event=None) -> None:
        """Rebuild self.calibration_data.event_views entirely from the per-image calibration
        layer's current contents - a full rebuild rather than tracking individual adds/removes,
        same reasoning as the generic-template sync: cheap given the realistic number of stamps,
        and it avoids the per-point diffing bugs the measurement layer needed real work to get
        right elsewhere in this refactor. This also naturally handles deletions, since there's
        no explicit 'remove a stamp' menu action - a student can only delete one via napari's
        own selection tool, which still fires this same event.
        """
        layer = self.event_calibration_layer()
        self.calibration_data.event_views = {}
        labels = layer.properties.get("labels", [])
        for i, point in enumerate(layer.data):
            if i >= len(labels):
                continue
            name = labels[i]
            if name not in FIDUCIAL_NAMES:
                continue
            view = int(point[0])
            evt = int(point[1])
            xy = [float(point[2]), float(point[3])]
            self.calibration_data.stamp(evt, view, name, xy)
        self.parent._sync_calibration_rows_into_table()
        self.parent.set_button_availability()

        print("event_views now:", self.calibration_data.event_views)

    def _setup_callbacks(self):
        for layer in self.generic_calibration_layers():
            # This is the callback to allow right-click on generic fiducials:
            layer.mouse_drag_callbacks.append(self.on_mouse)
            make_move_only(layer)
            # Keep every event's duplicate row of a dragged fiducial in sync with each other -
            # generic templates are event-independent by design (same physical mark regardless of
            # which photo you're looking at), but view-independent they are NOT (each camera
            # calibrates separately), so this only ever propagates within the same view.
            # Order matters: propagation must run first so every event's row is updated before
            # sync reads the event_index=0 representative row - otherwise a drag made while
            # viewing a non-zero event would sync a stale, not-yet-propagated position.
            layer.events.data.connect(self._propagate_generic_drag_across_events)
            layer.events.data.connect(self._sync_generic_template_from_layer)

        # This is the thing that changes which fiducials are visible when the view slider is slid:
        self.viewer.dims.events.current_step.connect(self.callback_calibration_layer_visibility)

        # symbol size management
        self.viewer.camera.events.zoom.connect(self.callback_symbol_size)

    def _num_events_on_generic_layer(self, layer) -> int:
        # 18 = 3 views * 6 slots, fixed by construction (see _generic_layer_row_index) - the
        # layer always holds exactly that many rows per event, so this is an exact division.
        return len(layer.data) // 18

    def _propagate_generic_drag_across_events(self, event=None) -> None:
        """When a generic fiducial is dragged, apply its new position to that same (view, slot)
        fiducial's duplicate row in every OTHER event, but not to any other view - each camera
        calibrates independently. Does not touch calibration_data (see the note in _setup_callbacks
        - that sync is fixed properly in 12-2c).
        """
        if event is None or event.action != "changed":
            return
        if getattr(self, "_propagating_generic_drag", False):
            return  # the .data reassignment below would otherwise re-trigger this same callback

        layer = self.generic_calibration_layers()[0]
        num_events = self._num_events_on_generic_layer(layer)
        if num_events <= 1:
            return  # nothing else to propagate to

        self._propagating_generic_drag = True
        try:
            data = layer.data
            for idx in event.data_indices:
                view, slot, event_index = self._generic_layer_view_slot_event(idx, num_events)
                new_y, new_x = data[idx][2], data[idx][3]
                for other_event_index in range(num_events):
                    if other_event_index == event_index:
                        continue
                    other_row = self._generic_layer_row_index(view, slot, other_event_index, num_events)
                    data[other_row][2] = new_y
                    data[other_row][3] = new_x
            layer.data = data
        finally:
            self._propagating_generic_drag = False

    def _sync_generic_template_from_layer(self, event=None) -> None:
        """Mirror the merged generic calibration layer's current fiducial positions into
        self.calibration_data - a full resync from the layer's own .data/.properties, not
        a surgical per-point patch, matching every other calibration sync in this file. Only the
        event_index=0 duplicate row of each (view, slot) is read - drag propagation keeps every
        event's copy identical, so any one representative row is sufficient, and generic_templates
        itself has no per-event concept at all.
        """
        layer = self.generic_calibration_layers()[0]
        num_events = self._num_events_on_generic_layer(layer)
        if num_events == 0:
            return

        for template in self.calibration_data.generic_templates:
            template.positions = {}
            template.slot_indices = {}

        labels = layer.properties.get("labels", [])
        data = layer.data
        for view in range(3):
            template = self.calibration_data.generic_templates[view]
            for slot in range(6):
                row = self._generic_layer_row_index(view, slot, 0, num_events)
                if row >= len(labels):
                    continue
                name = labels[row]
                if name in FIDUCIAL_NAMES:
                    y, x = data[row][2], data[row][3]
                    template.set_position(name, [float(y), float(x)], slot_index=slot)

    # TODO: could make generic_calibration_layers subservient to generic_calibration_layer_names instead of current way round.
    def generic_calibration_layer_names(self):
        return [layer.name for layer in self.generic_calibration_layers()]

    # TODO: could make generic_calibration_layers subservient to generic_calibration_layer_names instead of current way round.
    def generic_calibration_layers(self):
        return self._generic_calibration_layers

    def all_calibration_layers(self):
        # A simple python list of napari points layers.
        return self.generic_calibration_layers() + [ self.event_calibration_layer() ]

    # Callback for when the 'View' slider changes:
    def callback_calibration_layer_visibility(self, event):
        self._refresh_visibility_and_focus_of_all_calibration_layers()

    def callback_symbol_size(self, event: napari.utils.events.Event):
        screen_pixels_per_data_pixel = event.value # This value (up to zoom changes since call)
        # should be the same as self.viewer.camera.zoom.
        # Since that is what the refresh_symbol_sizes method uses as its default, both of
        # the following should be equal for all practical puropses:

        # Alternative one:
        # self.refresh_symbol_sizes(screen_pixels_per_data_pixel)
        # Alternative two:
        self.refresh_symbol_sizes()

    def refresh_symbol_sizes(self, screen_pixels_per_data_pixel=None):
        # https://napari.org/dev/guides/events_reference.html says that
        # event.value is "Scale from canvas pixels to world pixels." which is
        # not very clear. Experiment seems to clarify that it is "data pixel width" / "screen pixel width"
        # which (be careful here!) has units of "screen_pixels_per_data_pixel".
        # rather than the reciprocal of this.

        # I would like symbols for fiducials to typically be a fixed number of screen pixels in height, so that
        # they remain easy to see even when you zoom out.
        # If you zoom in far enough you are probably trying to place them precisely, so in that case I may wish
        # them to shrink a bit for fine placement, but this might not be necessary.
        if screen_pixels_per_data_pixel == None:
            screen_pixels_per_data_pixel = self.viewer.camera.zoom

        symbol_sizes_as_fractions_of_generic_symbol_size = {
            "front" : 1.0,
            "back" : 0.7,
            "point" : 0.5,
            "" : 1.0, # Generic or no-name
            None : 1.0 # Generic or no-name
        }

        generic_symbol_size_in_screen_pixels = 20

        symbol_sizes_in_screen_pixels = {
            key : val*generic_symbol_size_in_screen_pixels
            for key, val in symbol_sizes_as_fractions_of_generic_symbol_size.items()
        }

        symbol_sizes_in_data_pixels = {
            key : val/screen_pixels_per_data_pixel
            for key, val in symbol_sizes_in_screen_pixels.items()
        }

        def data_pixel_size_for(typ: str):
            if typ in symbol_sizes_in_data_pixels:
                return symbol_sizes_in_data_pixels[typ]
            else:
                return symbol_sizes_in_data_pixels[None] # Generic or no-name

        for layer in self.all_calibration_layers():
            orig_symbol_sizes = layer.size.copy()
            if "types" in layer.properties:
                types = layer.properties["types"]
            else:
                types = [None,] * len(layer.data)  # Fallback for when types are not supplied.
            new_symbol_sizes = [ data_pixel_size_for(typ)  for typ in types ]
            layer.size = new_symbol_sizes # This updates the symbol size as desired.

    # Show or hide calibration layers
    def set_calibration_layer_visibility_and_focus(self, visbility: bool, focus: bool):
        # visibility=True means that the correct view will be rendered and the others hidden, otherwise none will be shown.
        # focus=True means that when the relevant view is made visible, it will also be given focus.
        self._calibration_layer_visibility = visbility
        self._calibration_layer_focus = focus
        self._refresh_visibility_and_focus_of_all_calibration_layers()

    # Private method to make all the calibration layers invisible:
    def _hide_generic_calibration_layers(self):
        for layer in self.generic_calibration_layers():
            if layer.visible != False: # Avoid generating unnecessary triggers:
                layer.visible = False

    # Make the correct calibration layers visible/invisible based on the view slider:
    # Show/activate the (single, merged) generic calibration layer. Per-view switching is no
    # longer needed here at all - the layer's own per-point [view, event, y, x] coordinates
    # already make napari's normal dims-slicing show exactly the right 6 markers for whichever
    # view/event is current, the same way every other 4D layer in this plugin already works.
    def _show_and_activate_correct_generic_calibration_layer(self):
        layer = self.generic_calibration_layers()[0]

        if self._calibration_layer_focus:
            if self.viewer.layers.selection.active != layer:  # Avoid generating unnecessary triggers
                self.viewer.layers.selection.active = layer

        if not layer.visible:  # Avoid generating unnecessary triggers
            layer.visible = True
            layer.mode = "select"

    def _refresh_visibility_and_focus_of_all_calibration_layers(self):
        if self._calibration_layer_visibility:
            self._show_and_activate_correct_generic_calibration_layer()
            event_layer = self.event_calibration_layer()
            if not event_layer.visible:
                event_layer.visible = True
                # First-show default being select mode is most useful here.
                event_layer.mode = "select"
        else:
            self._hide_generic_calibration_layers()
            # Same guard _hide_generic_calibration_layers() already uses on the other layers,
            # and for the same reason - a redundant assignment still fires a real napari event,
            # which can crash if it happens while an image is still mid-insertion (the exact
            # class of bug fixed once already).
            event_layer = self.event_calibration_layer()
            if event_layer.visible != False:
                event_layer.visible = False

    def clone_only_this_fid_view_into_event(self, idx, name, generic_calibration_layer):
        # print(f"About to clone generic fiducial {idx=} with {name=}")
        destination_layer = self.event_calibration_layer()

        # TODO: Either current_event should be passed in (like view) or view should use current_step look up.
        # It makes no sense for one to do one and the other the other!
        current_event = self.viewer.dims.current_step[
            1]  # axis 0 is 'View', 1 is 'Event', 2 and 3 are image row and col

        # With one merged layer (12-2), "which view" can no longer be found by matching which
        # layer object was clicked - read it straight off the clicked point's own stored
        # coordinate instead. Robust regardless of the merged array's internal row layout.
        view = int(generic_calibration_layer.data[idx][0])

        # Don't allow unnamed fid insertion:
        if name == "" or name == None:
            napari.utils.notifications.show_error(f'Fiducial must have a name before it can be cloned.')
            return

        # Check that there is not already a fid with this name, and forbid injection if there is.
        # We do not want to have more than one fid with the same name in the event layer at a given view.
        mask = ((destination_layer.data[:, 0] == view) &
                (destination_layer.data[:, 1] == current_event) &
                (destination_layer.properties["labels"] == name))
        #print(f"Search for {ma,e} Saw {mask=} when \ndestination_layer.properties['labels'] was {destination_layer.properties['labels']} "
        #      f"and \ndestination_layer.data was {destination_layer.data}")

        if mask.any():
            napari.utils.notifications.show_error(f'There is already a fiducial named "{name}" in '
                                                  f"camera {view+1}'s view of event {current_event}.")
            return

        # The clicked point's data is now [view, event, y, x] (4D, since the merge) rather than
        # [y, x] - take only the last two.
        xy = generic_calibration_layer.data[idx][2:]
        label = generic_calibration_layer.properties["labels"][idx]
        #print(f"properties were {generic_calibration_layer.properties["labels"]}")
        #print(f"Found label {label=} in clone_fid_into_event for {view=}") # Correct label is being found, but wrong one stored.

        # Extend xy coords to 4D by adding view and event:
        fiducial_coords_4d_for_this_fiducial_in_view = [view, current_event, xy[0], xy[1]]

        # destination_layer.add() combined with current_properties has a confirmed napari quirk
        # here: adding one new point silently overwrites EVERY existing row's "labels" property
        # too, not just the new row's. Verified directly - printing labels immediately before and
        # after .add() showed already-correct earlier labels get corrupted the moment a second point
        # is added. Sidestepping it entirely: build the full new data/properties arrays ourselves and
        # assign them wholesale, rather than trusting .add() + current_properties to do it incrementally.
        old_data = destination_layer.data
        old_labels = list(destination_layer.properties.get("labels", []))
        new_point = np.array([fiducial_coords_4d_for_this_fiducial_in_view])
        new_data = np.vstack([old_data, new_point]) if len(old_data) else new_point
        new_labels = old_labels + [label]

        destination_layer.data = new_data
        destination_layer.properties = {"labels": np.array(new_labels, dtype=object)}
        destination_layer.current_symbol = "disc"
        # events.data fires the instant .data is assigned above, before the .properties line
        # even runs - so our connected sync callback would see the new point with a still-stale
        # labels array, skip it, and only pick it up on the NEXT stamp. Call it again explicitly,
        # now that both assignments are actually done, to guarantee correctness.
        self._sync_calibration_data_from_event_layer()

        destination_layer.text = destination_layer.text # Needed to get layer.text to become "aware" of property changes
        self.refresh_symbol_sizes()
        destination_layer.refresh() # render

    def clone_all_views_of_this_fid_into_event(self, idx, name):
        # print(f"About to clone generic fiducial {idx=} with {name=} into event.")
        # With one merged layer (12-2), "the same fiducial in another view" is a different row,
        # not the same idx on a different layer object - find each view's row for this same
        # logical (slot, event) and clone each one in turn.
        layer = self.generic_calibration_layers()[0]
        num_events = self._num_events_on_generic_layer(layer)
        _, slot, event_index = self._generic_layer_view_slot_event(idx, num_events)
        for view in range(3):
            other_idx = self._generic_layer_row_index(view, slot, event_index, num_events)
            self.clone_only_this_fid_view_into_event(other_idx, name, layer)

    def rename_point(self, idx, name, type):
        """Rename a generic fiducial. Two independent kinds of propagation happen here: within
        the clicked view, the front/back partner slot gets the paired name (e.g. naming a front
        slot "A'" also names its back partner "A") - unchanged logic from before the merge, just
        computed from a local slot number now rather than a raw array index. Separately, the same
        logical slot gets renamed in every view and every event too, since a name identifies one
        physical mark regardless of camera angle or which photo is showing.
        """
        layer = self.generic_calibration_layers()[0]
        num_events = self._num_events_on_generic_layer(layer)
        clicked_view, clicked_slot, clicked_event = self._generic_layer_view_slot_event(idx, num_events)

        # A name identifies one physical fiducial mark, so it must stay unique across slots.
        # Checking by slot, not by raw row, since the SAME slot's own name correctly appears on
        # every view/event's duplicate row already.
        if name != "":
            for row_idx, label in enumerate(layer.properties["labels"]):
                if label == name:
                    _, existing_slot, _ = self._generic_layer_view_slot_event(row_idx, num_events)
                    if existing_slot != clicked_slot:
                        napari.utils.notifications.show_error(
                            f'A generic fiducial named "{name}" already exists.'
                        )
                        return

        other_slot = None
        other_name = None
        if type == "front":
            other_slot = clicked_slot + 1  # we store front-then-back, so back is +1 on.
            other_name = name[:-1]  # all bar the last character (to remove prime)
        if type == "back":
            other_slot = clicked_slot - 1  # we store front-then-back, so front is -1 on.
            other_name = name + "'"

        slots_and_names = [(clicked_slot, name)]
        if other_slot is not None and other_name is not None:
            slots_and_names.append((other_slot, other_name))

        for slot, slot_name in slots_and_names:
            for view in range(3):
                for event_index in range(num_events):
                    row = self._generic_layer_row_index(view, slot, event_index, num_events)
                    layer.properties["labels"][row] = slot_name  # needed for saving purposes

        layer.text = layer.text  # necessary so that layer.text becomes "aware" of the changes we made to layer.properties
        layer.refresh()  # render changes to screen
        # Renaming mutates properties directly, which doesn't fire events.data on its own - sync
        # explicitly here, same reasoning as the original per-view version of this sync.
        self._sync_generic_template_from_layer()

    def on_mouse(self, layer, event):
        # This implements a right-click drop-down menu in response to a point in a generic calibration layer.
        # Note that on mac CTRL-left-click is a synonym for vanilla right-click, so don't expect to be able to use
        # CTRL as a modifier for left-click!  Note that add-to-selection in mac is CMD-left-click, so no
        # conflict with that.

        # Mouse events are defined in here https://github.com/vispy/vispy/blob/main/vispy/app/canvas.py

        """
        A traceback during a mouse event

      (venv) Gorfrog-MacWheird:cavendish-particle-tracks lester$   File "/Users/lester/github/cavendish-particle-tracks/./launch_debug.py", line 25, in <module>
    run()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_qt/qt_event_loop.py", line 469, in run
    app.exec_()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 626, in event
    out = super(QtBaseCanvasBackend, self).event(ev)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 496, in mousePressEvent
    self._vispy_mouse_press(
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/base.py", line 184, in _vispy_mouse_press
    ev = self._vispy_canvas.events.mouse_press(**kwargs)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 453, in __call__
    self._invoke_callback(cb, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 469, in _invoke_callback
    cb(event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 470, in _on_mouse_press
    self._process_mouse_event(mouse_press_callbacks, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 413, in _process_mouse_event
    mouse_callbacks(self.viewer, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/utils/interactions.py", line 125, in mouse_press_callbacks
    gen = mouse_drag_func(obj, event)
  File "/var/folders/xh/gkx93pyn2xl5jh11l4xwgcgr0000gn/T/ipykernel_68550/2820818860.py", line 5, in store
    traceback.print_stack()





     Here is another traceback during a mouse press handler that YIELDS after processing the press, and then receives a RELEASE:


     (venv) Gorfrog-MacWheird:cavendish-particle-tracks lester$ HELLO i=0
  File "/Users/lester/github/cavendish-particle-tracks/./launch_debug.py", line 25, in <module>
    run()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_qt/qt_event_loop.py", line 469, in run
    app.exec_()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 626, in event
    out = super(QtBaseCanvasBackend, self).event(ev)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 496, in mousePressEvent
    self._vispy_mouse_press(
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/base.py", line 184, in _vispy_mouse_press
    ev = self._vispy_canvas.events.mouse_press(**kwargs)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 453, in __call__
    self._invoke_callback(cb, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 469, in _invoke_callback
    cb(event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 470, in _on_mouse_press
    self._process_mouse_event(mouse_press_callbacks, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 413, in _process_mouse_event
    mouse_callbacks(self.viewer, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/utils/interactions.py", line 129, in mouse_press_callbacks
    next(gen)
  File "/var/folders/xh/gkx93pyn2xl5jh11l4xwgcgr0000gn/T/ipykernel_68550/740299691.py", line 7, in store
    traceback.print_stack()
Before event.type='mouse_press' event.button=2
yielding
HELLO i=1
  File "/Users/lester/github/cavendish-particle-tracks/./launch_debug.py", line 25, in <module>
    run()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_qt/qt_event_loop.py", line 469, in run
    app.exec_()
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 626, in event
    out = super(QtBaseCanvasBackend, self).event(ev)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/backends/_qt.py", line 506, in mouseReleaseEvent
    self._vispy_mouse_release(
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/app/base.py", line 224, in _vispy_mouse_release
    ev = self._vispy_canvas.events.mouse_release(**kwargs)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 453, in __call__
    self._invoke_callback(cb, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/vispy/util/event.py", line 469, in _invoke_callback
    cb(event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 484, in _on_mouse_release
    self._process_mouse_event(mouse_release_callbacks, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/_vispy/canvas.py", line 413, in _process_mouse_event
    mouse_callbacks(self.viewer, event)
  File "/Users/lester/github/cavendish-particle-tracks/venv/lib/python3.12/site-packages/napari/utils/interactions.py", line 216, in mouse_release_callbacks
    next(gen)
  File "/var/folders/xh/gkx93pyn2xl5jh11l4xwgcgr0000gn/T/ipykernel_68550/740299691.py", line 7, in store
    traceback.print_stack()
After event.type='mouse_release' event.button=2

        """
        our_sort_of_event = event.type == "mouse_press" and event.button == 2

        if not our_sort_of_event:
            return

        assert our_sort_of_event

        # Record position at mouse down (not at mouse up which could be different if there was a drag inbetween):
        coords = layer.world_to_data(event.position)

        # Wait for release:
        while event.type != "mouse_release":
            # use of yield explained in https://forum.image.sc/t/custom-mouse-shortcuts-to-help-creating-labels-in-napari/70930/6
            # There is also a sideways reference in https://napari.org/0.4.18/gallery/mouse_drag_callback.html# and another in
            # https://github.com/napari/napari/issues/3246#issuecomment-905803916 which mentions a generator being expected for the callback.
            # Perhaps this is the main documentation by example (but it does not have a mouse release): https://github.com/napari/napari/blob/main/examples/mouse_drag_callback.py
            yield
        assert event.type == "mouse_release"

        # We can now make the menu appear:

        index_of_nearest_point = layer.get_value(coords, world=True)
        if index_of_nearest_point is None:
            # This was not a right-click on a point.
            event.handled = True
            return

        i = index_of_nearest_point  # Just abbreviation shorthand.

        type = layer.properties["types"][i]
        name = layer.properties["labels"][i]

        def show_drop_down_menu(type):
            # build popup menu
            menu = QMenu(self.viewer.window._qt_window)

            # Choose one:
            THING = "name"
            #THING = "label"

            type_is_fiducial = type in ["front", "back"]

            if type_is_fiducial:
                header = QAction("Fiducial actions:", menu)
            else:
                header = QAction("Calibration point actions:", menu)

            header.setEnabled(False)  # makes it unclickable
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            menu.addAction(header)
            menu.addSeparator()

            from .analysis import FIDUCIAL_FRONT, FIDUCIAL_BACK

            if type == "front":
                fixed_names = sorted(FIDUCIAL_FRONT.keys())
            elif type == "back":
                fixed_names = sorted(FIDUCIAL_BACK.keys())
            else:
                fixed_names = ["origin", "decay", ]

            # add fixed names
            for fname in fixed_names:
                act = QAction(f'Set {THING} to "{fname}"', menu)
                act.triggered.connect(lambda _, f=fname: self.rename_point(i, f, type))
                menu.addAction(act)

            if type_is_fiducial:
                menu.addSeparator()
                clone_into_current_image_menu_item = QAction("Insert ONLY THIS VIEW of this fiducial into current event ...", menu)
                clone_into_current_image_menu_item.triggered.connect(
                    lambda _: self.clone_only_this_fid_view_into_event(i, name, layer))
                menu.addAction(clone_into_current_image_menu_item)

                clone_into_current_image_menu_item = QAction("Insert ALL VIEWS OF this fiducial into current event ...", menu)
                clone_into_current_image_menu_item.triggered.connect(lambda _: self.clone_all_views_of_this_fid_into_event(i, name))
                menu.addAction(clone_into_current_image_menu_item)

            # popup at cursor position
            menu.exec_(event.native.globalPos())

        show_drop_down_menu(type)
        event.handled = True

    def _default_generic_layer_arrays(self):
        """The default label/colour/type/symbol layout shared by every generic calibration layer,
        plus each view's own default point positions. Factored out so a brand-new session and restoring
        a saved one (filling in defaults for anything the save file didn't cover) share one copy
        of this layout, instead of risking two copies drifting apart.
        """
        from .analysis import TYPICAL_IMAGE_LONG_SIZE_PIX, TYPICAL_IMAGE_SHORT_SIZE_PIX

        origin_x = 0.5 * TYPICAL_IMAGE_SHORT_SIZE_PIX  # actually how far DOWN !!
        spread_x = 0.15 * TYPICAL_IMAGE_SHORT_SIZE_PIX  # actually vertical spread !!

        fid_step_y = 0.12 * TYPICAL_IMAGE_LONG_SIZE_PIX
        # Centre the whole span of fiducial pairs on the image's middle, rather than starting AT
        # the middle and spreading downward from there - that offset used to leave room above for
        # the now-removed origin/decay points, which no longer applies. Computed from the actual
        # pair count so it stays correct if that number is ever changed.
        fid_origin_y = (
                0.5 * TYPICAL_IMAGE_LONG_SIZE_PIX
                - 0.5 * (CalibrationManager.num_generic_front_back_fid_pairs - 1) * fid_step_y
        )

        # Front/Back fiducial pairs (the old "origin"/"decay" measurement points that used to
        # live here are gone - vestigial since Step 4 moved live origin/decay/track measurement
        # onto the "Radii and Lengths" layer; the menu actions and table-cloning methods that
        # only ever applied to them have been removed too):
        labels = []
        symbols = []
        colours = []
        types = []
        points_in_generic_view = []

        # Now position the Front/Back fiducial pairs:
        for i in range(CalibrationManager.num_generic_front_back_fid_pairs):
            labels += ["", "", ]
            types += ["front", "back", ]
            symbols += ["x", "x", ]
            points_in_generic_view += [
                [origin_x - spread_x, fid_origin_y + i * fid_step_y, ],
                [origin_x + spread_x, fid_origin_y + i * fid_step_y, ],
            ]
            if i == 0:
                colours += [
                    "#55ff00",  # front fiducial (light green)
                    "#00aa00",  # back fiducial (dark green)
                ]
            elif i == 1:
                colours += [
                    "#ff5500",  # front fiducial (light red)
                    "#aa0000",  # back fiducial (dark red)
                ]
            else:
                colours += [
                    "#5500ff",  # front fiducial (light blue)
                    "#0000aa",  # back fiducial (dark blue)
                ]

        # Displace the generic points 100 to the left, or not at all, or 100 to the right, depending on view:
        points_in_view = [
            np.array([np.array(point) + np.array([(v - 1) * 100, 0, ]) for point in points_in_generic_view])
            for v in view_indices
        ]

        return points_in_view, labels, colours, types, symbols

    def _generic_layer_row_index(self, view, slot, event_index, num_events):
        """Canonical mapping from a logical fiducial slot to a row in the merged generic layer's
        data. Rows are grouped as (view, slot) blocks of num_events contiguous rows each - one real
        duplicate row per event, so a fiducial correctly shows up regardless of which event is being
        viewed. The old 3-separate-layers system got this "for free" by having no event axis at all;
        a real 4D layer needs it done explicitly. slot runs 0-5 (front,back alternating within a view,
        matching today's order).
        """
        return (view * 6 + slot) * num_events + event_index

    def _generic_layer_view_slot_event(self, idx, num_events):
        """Inverse of _generic_layer_row_index."""
        event_index = idx % num_events
        block = idx // num_events
        view = block // 6
        slot = block % 6
        return view, slot, event_index

    def _merged_generic_layer_arrays(self, num_events):
        """Combine the per-view arrays from _default_generic_layer_arrays() (left untouched,
        since restore logic still depends on it until 12-2d) into one merged layout with num_events
        duplicate rows per logical (view, slot) fiducial - see _generic_layer_row_index for
        the exact row layout. Points become 4D [view, event, y, x] coordinates, matching every other
        multi-dimensional layer in this plugin.
        """
        points_in_view, labels, colours, types, symbols = self._default_generic_layer_arrays()

        total_rows = 3 * 6 * num_events
        merged_points = [None] * total_rows
        merged_labels = [None] * total_rows
        merged_colours = [None] * total_rows
        merged_types = [None] * total_rows
        merged_symbols = [None] * total_rows

        for view in range(3):
            for slot in range(6):
                y, x = points_in_view[view][slot]
                for event_index in range(num_events):
                    row = self._generic_layer_row_index(view, slot, event_index, num_events)
                    merged_points[row] = [view, event_index, y, x]
                    merged_labels[row] = labels[slot]
                    merged_colours[row] = colours[slot]
                    merged_types[row] = types[slot]
                    merged_symbols[row] = symbols[slot]

        return np.array(merged_points), merged_labels, merged_colours, merged_types, merged_symbols

    def _default_generic_calibration_layers(self):
        # Placeholder single-event count - the real count isn't knowable yet at this point in the
        # normal flow (CalibrationManager is constructed before images are loaded). Rebuilt for
        # real via rebuild_generic_layer_for_event_count() once images actually load.
        points, labels, colours, types, symbols = self._merged_generic_layer_arrays(num_events=1)

        # If in debug mode can replace the points and labels with ones that are physically interesting.
        # Don't give this option to the students!
        debug_fiducial_mode = False

        if debug_fiducial_mode and CalibrationManager.num_generic_front_back_fid_pairs == 3:
            # TODO: the debug point sets predate the merged-layer layout (12-2) and are 2D
            # per-view arrays - they'd need updating to the merged [view, event, y, x] shape
            # before this branch can be used again. Left disabled rather than silently wrong.
            pass

        return [self._single_generic_configuration_layer(points, labels, colours, types, symbols)]

    def rebuild_generic_layer_for_event_count(self, num_events) -> None:
        """Rebuild the merged generic calibration layer so each of its 18 logical fiducial slots has
        one real duplicate row per event, replacing the num_events=1 placeholder it was built with at
        __init__ time. Called once, right after the real image data is loaded and the true event count
        is finally known.
        """
        points, labels, colours, types, symbols = self._merged_generic_layer_arrays(num_events)
        layer = self.generic_calibration_layers()[0]
        layer.data = points
        # border_color/face_color/symbol are separate from properties and don't automatically
        # resize themselves to match a longer .data array - reassign them explicitly too, rather
        # than risk a length mismatch (the exact class of bug properties dict rebuilds bit us with
        # once already, in 11b-ii).
        layer.border_color = colours
        layer.face_color = colours
        layer.symbol = symbols
        layer.properties = {
            "labels": np.array(labels, dtype=object),
            "colours": colours,
            "types": types,
            "symbols": symbols,
        }
        # Rebuilding text properly, not just reassigning to itself.
        layer.text = {
            'string': 'labels',
            'color': colours,
            'size': 12,
            'anchor': 'center',
            'translation': np.array([0, 0, -150, 0]),
        }
        layer.refresh()
        # mark_clean() was called once already, at __init__ - before this rebuild, and before the
        # real event count was even knowable. Without re-marking clean here, the dirty-check
        # baseline stays frozen at that placeholder shape forever, and every future dirty_things()
        # call crashes trying to compare against it (shape mismatch.
        self.mark_clean()

    def _restore_generic_calibration_layers(self, generic_templates) -> None:
        """Rebuild the merged generic calibration layer from a loaded session's saved fiducial
        positions, filling in the standard default position/label for any slot the save file didn't
        cover. The layer is move-only (no add/delete), so a name missing entirely after a load would
        otherwise be permanently unplaceable for the rest of the session.

        Each name is put back in its own original slot (via slot_indices) wherever possible, so colour
        and relative layout next to any still-unlabelled slots both survive a round-trip unchanged.
        Falls back to claiming the first available slot of the matching type only for names with no
        recorded slot (e.g. a .pkl saved before slot tracking existed) or a slot collision.

        Since generic_templates has no per-event concept at all, the resolved position for each
        (view, slot) is replicated across every event's duplicate row - num_events comes from the
        currently loaded layer's own row count, not from anything in the save file, so a session
        saved against a different event count is handled correctly automatically.
        """
        from .analysis import FIDUCIAL_FRONT, FIDUCIAL_BACK

        layer = self.generic_calibration_layers()[0]
        num_events = self._num_events_on_generic_layer(layer)
        if num_events == 0:
            return

        points_in_view, default_labels, colours_base, types_base, symbols_base = self._default_generic_layer_arrays()

        total_rows = 3 * 6 * num_events
        merged_points = [None] * total_rows
        merged_labels = [None] * total_rows
        merged_colours = [None] * total_rows
        merged_types = [None] * total_rows
        merged_symbols = [None] * total_rows

        for view_index in range(3):
            template = generic_templates[view_index]
            labels = list(default_labels)
            points = [list(p) for p in points_in_view[view_index]]

            assigned_slots = set()
            unresolved_names = []
            for name, xy in template.positions.items():
                slot_index = template.slot_indices.get(name)
                is_front = name in FIDUCIAL_FRONT
                is_back = name in FIDUCIAL_BACK
                slot_type_matches = slot_index is not None and 0 <= slot_index < len(types_base) and (
                        (is_front and types_base[slot_index] == "front")
                        or (is_back and types_base[slot_index] == "back")
                )
                if slot_type_matches and slot_index not in assigned_slots:
                    labels[slot_index] = name
                    points[slot_index] = xy
                    assigned_slots.add(slot_index)
                else:
                    unresolved_names.append(name)

            if unresolved_names:
                front_slots = [i for i, t in enumerate(types_base) if t == "front" and i not in assigned_slots]
                back_slots = [i for i, t in enumerate(types_base) if t == "back" and i not in assigned_slots]
                for name in sorted(unresolved_names):
                    target_slots = front_slots if name in FIDUCIAL_FRONT else back_slots
                    if not target_slots:
                        continue
                    slot_index = target_slots.pop(0)
                    labels[slot_index] = name
                    points[slot_index] = template.positions[name]

            # Replicate this view's resolved slots across every event's duplicate row.
            for slot in range(6):
                y, x = points[slot]
                for event_index in range(num_events):
                    row = self._generic_layer_row_index(view_index, slot, event_index, num_events)
                    merged_points[row] = [view_index, event_index, y, x]
                    merged_labels[row] = labels[slot]
                    merged_colours[row] = colours_base[slot]
                    merged_types[row] = types_base[slot]
                    merged_symbols[row] = symbols_base[slot]

        layer.data = np.array(merged_points)
        # border_color/face_color/symbol are separate from properties and don't automatically
        # resize themselves to match a longer .data array - the same lesson learned (the hard
        # way) in rebuild_generic_layer_for_event_count applies identically here.
        layer.border_color = merged_colours
        layer.face_color = merged_colours
        layer.symbol = merged_symbols
        layer.properties = {
            "labels": np.array(merged_labels, dtype=object),
            "colours": merged_colours,
            "types": merged_types,
            "symbols": merged_symbols,
        }
        # Rebuilt properly, not just layer.text = layer.text - that was still holding a stale,
        # wrong-length colour array in the exact bug we just fixed in the sibling rebuild method.
        layer.text = {
            'string': 'labels',
            'color': merged_colours,
            'size': 12,
            'anchor': 'center',
            'translation': np.array([0, 0, -150, 0]),
        }
        layer.refresh()

        self._sync_generic_template_from_layer()

    def _restore_event_calibration_layer(self) -> None:
        """Rebuild the per-image calibration layer's visible stamps from calibration_data.event_views
        (already restored by the data-side load). This layer holds every event's stamps at once, sliced
        by view/event for display, same as during normal live use.
        """
        layer = self.event_calibration_layer()
        points = []
        labels = []
        for (event, view), fiducial_view_data in self.calibration_data.event_views.items():
            for name, xy in fiducial_view_data.stamped.items():
                points.append([view, event, xy[0], xy[1]])
                labels.append(name)

        layer.data = np.array(points) if points else np.empty((0, 4))
        layer.properties = {"labels": np.array(labels, dtype=object)}
        layer.current_symbol = "disc"
        layer.text = layer.text
        layer.refresh()
        # events.data fires the instant .data is assigned above, before .properties has even run -
        # the same lag bug fixed once already elsewhere in this file.
        self._sync_calibration_data_from_event_layer()

    def _single_generic_configuration_layer(self, points, labels, colours, types, symbols):
        """
        The point of this function is to provide a single route through which the generic config layer
        is constructed, so that even if it needs internally derived settings, or things not in a csv file,
        they can be applied consistently. One merged layer now, not one per camera (12-2) - sliced by the
        View axis instead of requiring manual switching.
        """
        props = {
            'labels': labels,
            'colours': colours,
            'types': types,
            'symbols': symbols,
        }
        layer = napari.layers.Points(
            points,
            name=GENERIC_CALIBRATION_LAYER_NAME,
            ndim=4,
            properties=props,
            border_width=7,
            border_width_is_relative=False,
            border_color=colours,
            face_color=colours,
            symbol=symbols,
            visible=False,
        )
        layer.text = {
            'string': 'labels',  # This is a key in properties
            'color': colours,
            'size': 12,
            'anchor': 'center',
            'translation': np.array([0, 0, -150, 0]),
        }
        layer.refresh()
        return layer