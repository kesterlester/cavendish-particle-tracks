import pandas as pd
import numpy as np
import napari

def save_calib_points_to_csv(points_layer, filename):
    """Save a napari Points layer (coords + properties) to CSV."""
    coords = points_layer.data
    props = points_layer.properties
    
    # Start with coords
    df = pd.DataFrame(coords, columns=[f"axis_{i}" for i in range(coords.shape[1])])
    
    # Add props as extra columns
    for key, values in props.items():
        df[key] = values
    
    df.to_csv(filename, index=False)


def load_calib_points_from_csv(viewer, filename, name="loaded points"):
    """Load points (coords + properties) from CSV into a napari viewer."""
    df = pd.read_csv(filename)
    
    # Assume first D columns are coordinates, the rest are properties
    # (if you know D in advance you can make this explicit)
    coords = df.iloc[:, :2].to_numpy()  # adjust slice if 3D etc.
    props = df.iloc[:, 2:].to_dict(orient="list")
    
    return viewer.add_points(coords, properties=props, name=name)
