import pytest
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QMessageBox, QWidget

from cavendish_particle_tracks._main_widget import ParticleTracksWidget


class _KeepWindowsOffScreen(QObject):
    """Marks every top-level window WA_DontShowOnScreen just before it's mapped. Qt sends the
    Show event before creating the native OS window, so this catches everything - including
    dialogs built inside Qt's own C++ helpers (QFileDialog.getExistingDirectory etc.), which a
    Python-side patch of QWidget.show() never would. Qt still treats the widget as visible
    (isVisible(), layout, activeModalWidget(), QTest clicks all behave normally); it just never
    appears on the developer's screen.

    QT_QPA_PLATFORM=offscreen would be the usual one-line way to do this, but it segfaults here:
    napari's vispy canvas needs a real OpenGL context, which the offscreen/minimal platforms
    don't provide on macOS.
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow():
            obj.setAttribute(Qt.WA_DontShowOnScreen, True)
        return False


@pytest.fixture(autouse=True, scope="session")
def keep_test_windows_off_screen(qapp):
    event_filter = _KeepWindowsOffScreen()
    qapp.installEventFilter(event_filter)
    yield
    qapp.removeEventFilter(event_filter)


@pytest.fixture(autouse=True)
def auto_discard_unsaved_changes_popup(monkeypatch):
    """Closing a test's fake napari window can trigger the plugin's real
    'discard unsaved changes?' popup. Auto-answer it as Discard so tests
    don't sit there waiting on a click that's never coming."""
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Discard)


@pytest.fixture
def cpt_widget(make_napari_viewer):
    """Common test setup fixture: calls the napari helper fixture
    `make_napari_viewer` then creates the ParticleTracksWidget."""
    viewer = make_napari_viewer()
    widget = ParticleTracksWidget(napari_viewer=viewer)
    return widget

