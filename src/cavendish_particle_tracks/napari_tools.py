import napari
from qtpy.QtWidgets import QMessageBox

import time
import functools

def at_most_once_per(interval_in_seconds: float):
    """
    Decorator to ensure a function runs at most once per `interval` seconds.
    If called too soon, it returns None without executing.

    Usage:
    @at_most_once_per(2.0)  # allow once every 2 seconds
    def say_hi():
        print("hi")

    say_hi()  # prints
    say_hi()  # called within 2s -> dropped
    """
    def decorator(func):
        last_called = [0.0]  # use list as mutable container

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.monotonic()
            if now - last_called[0] >= interval_in_seconds:
                last_called[0] = now
                return func(*args, **kwargs)
            # Too soon: drop call silently
            return None
        return wrapper
    return decorator

from copy import deepcopy
def overwrite_layer(A, B):
    if type(A) is not type(B):
        raise TypeError("Layer types must match")

    data, meta, _ = B.as_layer_data_tuple()
    A.data = deepcopy(data)

    SETTABLE_KEYS = {
        "name", "metadata", "properties", "face_color", "edge_color",
        "size", "symbol", "edge_width", "opacity", "blending",
        "visible", "scale", "translate", "rotate"
    } # Original list.

    SETTABLE_KEYS = {
        # core
        "name", "metadata", "properties", "size", "symbol",
        "border_width", "border_width_is_relative",
        "face_color", "border_color",
        "opacity", "blending", "visible",
        "scale", "translate", "rotate", "shear", "affine",
        "projection_mode", "units",

        # optional but safe to include for full fidelity
        "axis_labels", "experimental_clipping_planes",
        "face_color_cycle", "face_colormap", "face_contrast_limits",
        "border_color_cycle", "border_colormap", "border_contrast_limits",
        "text", "out_of_slice_display", "n_dimensional",
        "features", "feature_defaults",
        "shading", "antialiasing", "canvas_size_limits", "shown",
    }

    for k in SETTABLE_KEYS:
        if k in meta:
            setattr(A, k, deepcopy(meta[k]))

def make_move_only(layer: napari.layers.Points) -> None:
    """Patch an existing napari Points layer to allow move-only editing."""

    @at_most_once_per(3) # seconds
    def explain_veto(layer, show_dialog=True):
        print(f"Showing GUI warning for add mode disabled on {layer.name}")
        if show_dialog:
            mes = QMessageBox()
            mes.setText(f"Adding or deleting points is not possible on the move-only layer '{layer.name}'.")
            mes.exec_()

    if layer._type_string != "points":
        raise TypeError("make_move_only only works on Points layers")

    vetoed_modes = [ "add", ]
    default_mode = "select"

    # Force to select mode at beginning
    layer.mode = default_mode

    # Prevent deletion via the ❌ button with this monkey patch:
    def no_delete():
        print("Deletion via GUI disabled")
        explain_veto(layer, show_dialog=True)
    layer.remove_selected = no_delete

    # Prevent mode switching into vetoed_modes
    def lock_mode(event):
        print(f"LM saw mode {layer.mode}")
        if layer.mode in vetoed_modes:
            print(f"LM vetoing")
            layer.mode = default_mode # it would be nicer to go back to previous mode, but don't know what it was.
            explain_veto(layer)
        else:
            print(f"LM not vetoing")

    layer.events.mode.connect(lock_mode)

    # Prevent deletion
    @layer.bind_key('Backspace') # point deletion
    @layer.bind_key('Delete') # point deletion
    @layer.bind_key('A') # add mode
    def prevent(layer):
        print(f"BIND_KEY veto delete/add")
        explain_veto(layer, show_dialog=True)

    return layer

