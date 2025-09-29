"""
The class NonOverlappingQDialog prevents dialogs opening at the same
time as others if they have incompatible needs.

Example usage:

class MagnificationDialog(NonOverlappingQDialog):
    def __init__(self, parent):
        super().__init__(tokens=["some resource", "some other resource"], parent=parent, title="Magnification")
        self.parent: ParticleTracksWidget = parent

class AnotherDialog(NonOverlappingQDialog):
    def __init__(self, parent):
        # forgot to set title → will default to class name "AnotherDialog"
        super().__init__(tokens=["some resource"], parent=parent)

If you open MagnificationDialog then try AnotherDialog, the warning will say:
"Can't open dialog 'AnotherDialog' until Magnification is closed."
"""

from qtpy.QtWidgets import QDialog, QMessageBox

class NonOverlappingQDialog(QDialog):
    # class-wide registry: token -> dialog instance
    _token_registry = {}

    def __init__(self, tokens, *args, title=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens = set(tokens)
        self._registered = False

        # Ensure a sensible title is always set
        effective_title = title or type(self).__name__
        self.setWindowTitle(effective_title)

    def open(self):
        # check for conflicts before showing
        conflicts = []
        for token in self._tokens:
            if token in NonOverlappingQDialog._token_registry:
                conflicts.append(NonOverlappingQDialog._token_registry[token])

        if conflicts:
            names = ", ".join(d.windowTitle() for d in conflicts)
            QMessageBox.warning(
                self,
                "Conflict",
                f"Can't open dialog '{self.windowTitle()}' "
                f"until {names} is closed."
            )
            return

        # no conflict: register tokens
        for token in self._tokens:
            NonOverlappingQDialog._token_registry[token] = self
        self._registered = True

        super().open()  # or self.show()

    def closeEvent(self, event):
        # clean up registry when dialog closes
        if self._registered:
            for token in self._tokens:
                NonOverlappingQDialog._token_registry.pop(token, None)
            self._registered = False
        super().closeEvent(event)
