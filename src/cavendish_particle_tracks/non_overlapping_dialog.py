"""
The class NonOverlappingQDialog prevents dialogs opening at the same
time as others if they have incompatible needs. Note that it's OPEN not CONSTRUCT that is prevented.
This is by design as you might want to persist the object but show/hide it without reconstructing it.
So resource allocation should be done within open.

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

    def _exit_without_calling_super(self):
        # check for conflicts before showing
        conflicts = []
        for token in self._tokens:
            if token in NonOverlappingQDialog._token_registry:
                conflicts.append(NonOverlappingQDialog._token_registry[token])

        if conflicts:
            conflicting_window_titles = [d.windowTitle() for d in conflicts]
            we_conflict_with_ourself = self.windowTitle() in conflicting_window_titles

            if we_conflict_with_ourself:
                QMessageBox.warning(
                    self,
                    "Conflict",
                    f"The '{self.windowTitle()}' "
                    f"window is already open."
                )
            else:
                names = ", ".join(conflicting_window_titles)
                QMessageBox.warning(
                    self,
                    "Conflict",
                    f"Can't open dialog '{self.windowTitle()}' "
                    f"until {names} is closed."
                )

            return True
            """ 
            Alternatively:

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Conflict")
            msg.setText(
                 f"Can't open dialog '{effective_title}' "
                 f"until {names} is closed."
            )
            msg.exec_()  # stays until user clicks OK
            """

        # no conflict: register tokens
        for token in self._tokens:
            NonOverlappingQDialog._token_registry[token] = self

        self._registered = True
        return False

    def show(self) -> None:
        if self._exit_without_calling_super():
            return
        super().show()

    def open(self) -> None:
        if self._exit_without_calling_super():
            return
        super().open()

    def closeEvent(self, event):
        # clean up registry when dialog closes
        if self._registered:
            for token in self._tokens:
                NonOverlappingQDialog._token_registry.pop(token, None)
            self._registered = False
        super().closeEvent(event)


