"""
The intercept_close decorator manages the intercepting of window close requests so that data can be saved.

Use like this:

##############################
from intercept_close import InterceptClose

class Foo:
    pass

@InterceptClose
class MyPluginWidget(Foo):  # Foo can be any base class
    # A clean plugin widget that doesn’t worry about close interception.

    def __init__(self, viewer):
        super().__init__() # initialise Foo
        self.viewer = viewer

    def has_unsaved_data(self):
        return True # or something more appropriate!

import napari
viewer = napari.Viewer()
plugin = MyPluginWidget(viewer)
napari.run()
##############################
"""

import functools
from qtpy.QtCore import QObject, QEvent
from qtpy.QtWidgets import QMessageBox

def InterceptClose(cls):
    """Class decorator to add close-interception behavior."""

    class Wrapped(cls):
        def __init__(self, viewer, *args, **kwargs):
            super().__init__(viewer, *args, **kwargs)
            self._dirty = False
            self._interceptor = _CloseInterceptor(viewer, self)
            viewer.window._qt_window.installEventFilter(self._interceptor)

    # copy metadata so it looks like the original class
    functools.update_wrapper(Wrapped, cls, updated=())
    Wrapped.__name__ = cls.__name__
    Wrapped.__qualname__ = cls.__qualname__
    Wrapped.__doc__ = cls.__doc__

    return Wrapped


class _CloseInterceptor(QObject):
    def __init__(self, viewer, plugin):
        super().__init__()
        self.plugin = plugin
        self.viewer = viewer

    def eventFilter(self, obj, event):

        if event.type() == QEvent.Close:
            if not getattr(self.plugin, "dirty_things"):
                print("Warning!  Did your plugin forget to implement the dirty_things method needed by the CLoseInterceptor?")

            dirty_things = getattr(self.plugin, "dirty_things", lambda: [])()
            if dirty_things:
                #reply = QMessageBox.warning(
                #    obj,
                #    f"Unsaved things: {dirty_things}",
                #    "Discard unsaved changes?",
                #    QMessageBox.Discard | QMessageBox.Cancel,
                #    QMessageBox.Cancel,
                #)
                msg = QMessageBox(obj)
                msg.setWindowTitle(f"Unsaved things: {dirty_things}")
                if len(dirty_things)==1:
                    msg.setText(f"Discard unsaved changes detected in {dirty_things[0]} ?")
                else:
                    msg.setText(f"Discard unsaved changes detected in {dirty_things} ?")
                msg.setIcon(QMessageBox.Warning)
                # Set the buttons and default button
                msg.setStandardButtons(QMessageBox.Discard | QMessageBox.Cancel)
                msg.setDefaultButton(QMessageBox.Cancel)
                # Show the box and get the result
                reply = msg.exec()

                if reply == QMessageBox.Cancel:
                    event.ignore()
                    return True # Also says event should be ignored.  See https://doc.qt.io/archives/qt-5.15/qobject.html#eventFilter
                # elif reply == QMessageBox.No:
                # Could instead do some plugin action, like
                # self.plugin.save_data()

        return super().eventFilter(obj, event)


