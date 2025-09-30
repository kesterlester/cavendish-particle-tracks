from intercept_close import InterceptClose

##############################

class Foo:
    pass
    
@InterceptClose
class MyPluginWidget(Foo):  # Foo can be any base class
    # A clean plugin widget that doesn’t worry about close interception.

    def __init__(self, viewer):
        super().__init__() # initialise Foo
        self.viewer = viewer

    def save_data(self):
        print("Saving my data...")

##############################

import napari
viewer = napari.Viewer()
plugin = MyPluginWidget(viewer)

plugin.mark_dirty(True)  # mark as having unsaved data
napari.run()
