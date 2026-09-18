# Architecture notes

*Last updated: September 2026, reflecting the codebase through v2.2.1 (git tag by ajalagekar).*

This document explains how the plugin is currently put together and why key parts are shaped
the way they are. It's aimed at whoever picks this codebase up next — `docs/user-manual.md`
covers how to *use* the tool; this covers how it *works*.

## Core data model (`analysis.py`)

- **`ParticleDecay`** — one row per measured process. Holds `name`, `index` (the process type),
  `event_number`, and `views`: a list of exactly 3 `ViewData` objects, one per camera.
- **`ViewData`** — one camera's measurements for one process: `origin`, `decay`, up to 3
  `track_points`, and (for the one process type with two charged daughters) `decay_angle_lines`.
  Derived values (`radius_px`, `length_px`, `phi_proton`, `phi_pion`) are **never set directly**
  — they're recomputed automatically inside `set_origin`/`set_decay`/`set_track_points`/
  `set_decay_angle_lines`. Anything reconstructing this data (e.g. the CSV loader) should only
  ever call these setters, never assign the derived fields by hand.
- **`CalibrationRow`** — one row per event with fiducial stamps. Mirrors `ParticleDecay`'s shape
  (`views`, a list of 3 `FiducialViewData`) so both row types can share table columns and CSV
  structure.
- **`FiducialViewData.stamped`** — a `{name: [x, y]}` dict of whichever fiducials were stamped
  in that view/event.
- **`GenericFiducialTemplate`** — the reusable per-camera calibration positions, one per view,
  event-independent. `slot_indices` records which physical slot each name occupies, so colour
  and layout survive a save/load round-trip.
- **`SavedSession`** — the top-level `.pkl` payload: `data` (list of `ParticleDecay`/
  `CalibrationRow`) plus `generic_templates`.

## Layer architecture (`_calibration_manager.py`, `_main_widget.py`)

Four interactive layers: `"Radii and Lengths"`, `"Calibration (generic)"`,
`"Calibration (per-image)"`, `"Decay Angles Tool"`.

The generic calibration layer is a single merged layer, not one per camera. It's genuinely 4D
(`[view, event, row, col]`) with a real duplicate row per event for every one of its 18 logical
fiducial slots (3 views × 6 slots). This was **not** obvious going in — napari right-aligns a
layer's axes to the viewer's *trailing* axes when the layer has fewer dimensions than the
viewer, so a naive 3D version silently misbehaves (see "Known gotchas" below). Row layout is
governed by `_generic_layer_row_index`/`_generic_layer_view_slot_event` — treat these as the
single source of truth for how rows map to (view, slot, event) before touching this layer.

Renaming propagates across all views *and* events (one name = one physical mark). Dragging
propagates across events only, never views (each camera calibrates independently).

## Navigation system (`_main_widget.py`)

Four buttons (`radii_lengths_nav_button`, `generic_fiducials_nav_button`,
`saved_fiducials_nav_button`, `decay_angles_nav_button`) replace direct interaction with
napari's native layer list, which is hidden by default (`_qt_viewer.dockLayerList`, see gotchas
below) but still recoverable via napari's own Window menu.

All four route through one shared method, `_activate_layer`, and one shared listener,
`_on_active_layer_changed`, which fires on *any* active-layer change regardless of cause (button,
keyboard shortcut, or someone re-showing the native list) and keeps both the button highlighting
and the decay-angle diagram's show/hide/reorder state correct as a pure function of "what's
active right now" — not tracked as a transition. Keyboard shortcuts are G/H/J/K, chosen after two
rounds of collisions with napari's own reserved per-layer-type shortcuts (bare digits and bare
letters are both partially claimed — see gotchas).

## CSV format (`analysis.py`: `CSV_COLUMNS`, `to_csv_rows`, `load_csv_session`)

Long format: every `ParticleDecay`/`CalibrationRow` writes **3 rows**, one per view, sharing a
`row_group_id` (assigned fresh at save time via `enumerate`, not a stored field — table position
isn't stable across deletions, so this must never be cached). Process-level fields repeat
identically across a group's 3 rows; per-view fields differ per row.

A second, small table (view/name/x/y/slot_index) holds the generic fiducial templates, since
they're workspace-level, not tied to any row. It's appended after a blank line in the same file.
Students split on `\n\n` and parse each chunk separately with `pd.read_csv` — a naive single
`read_csv` call does **not** stop at the blank line and will silently corrupt both tables.

Values are rounded (`round_px`, 1dp; `round_angle`, 4dp) for both CSV export and table display —
deliberately below any real measurement precision, so this loses nothing meaningful. The reload
path (`load_csv_session`) reconstructs objects using only the *raw* setters (never the derived
columns), so radius/length/angle are always recomputed fresh, guaranteed consistent with live
measurement.

`vars_to_save()`/`vars_to_show()` (the *live table's* column set) are deliberately **separate**
from `CSV_COLUMNS` — they drive real `QTableWidget` column lookups (`_get_table_column_index`)
throughout `_main_widget.py`, so redesigning the CSV format never risks the live table's
structure.

## Save/load format

`.pkl` support still exists in full but is switched off via `ENABLE_PICKLE = False` (top of
`_main_widget.py`, same pattern as the pre-existing `ENABLE_MAG` flag) — flip it back to `True`
to restore it; nothing else needs to change.

## Deliberately rejected designs

- **A true single layer for generic templates *and* per-event stamps.** Generic templates have
  no natural event axis; forcing one in would mean continuously rewriting 18 coordinates to
  chase whatever event is current — a synchronization burden with no existing precedent here.
- **Sparse per-fiducial-name CSV columns *as a packed string* instead of real columns.** Rejected
  in favour of genuine numeric columns — a packed string would need custom parsing before a
  student's script could use it at all, working against the format's whole purpose.
- **Per-layer-type keyboard shortcuts on bare digits or bare letters.** Both collide with napari's
  own built-in shortcuts (see below) — confirmed the hard way, twice.

## Known gotchas

- **napari axis alignment**: a layer with fewer dimensions than the viewer is right-aligned to
  the viewer's *trailing* axes, not its leading ones. Costly to discover; cheap to remember.
- **`viewer.window._qt_viewer`** (used to hide the native layer list) is a private,
  deprecated-with-warning napari API. Wrapped in `try/except AttributeError` everywhere it's
  used — worst case on a napari upgrade is the list just stays visible, never a crash.
- **`QHeaderView.ResizeToContents`** recalculates continuously from whatever content exists *at
  that moment* — it cannot be given a "starting width" that survives past the first real content
  change. Relevant if `saved_vertices`' initial width is ever revisited.
- **napari reserved keybindings**: bare `S` is Points-layer "select mode"; Shapes layers reserve
  several more. Any future shortcut work should search napari's actual `_key_bindings.py` files
  before picking a key, not assume a letter is free.
