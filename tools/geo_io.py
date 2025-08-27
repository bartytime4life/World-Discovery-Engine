# tools/geo_io.py
# Lightweight, lazy GIS I/O utilities with soft dependencies.
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict

def _lazy_imports():
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.windows import Window
        from shapely.geometry import box, Point, shape
    except Exception as e:
        gpd = None
        rasterio = None
        Window = None
        box = Point = shape = None
    return gpd, rasterio, Window, box, Point, shape

def read_vector(path: str | Path):
    gpd, *_ = _lazy_imports()
    if gpd is None:
        raise RuntimeError("geopandas is unavailable in this environment.")
    return gpd.read_file(str(path))

def read_raster_window(path: str | Path, x: int, y: int, w: int, h: int):
    _, rasterio, Window, *_ = _lazy_imports()
    if rasterio is None or Window is None:
        raise RuntimeError("rasterio/window ops unavailable in this environment.")
    with rasterio.open(str(path)) as ds:
        window = Window(col_off=x, row_off=y, width=w, height=h)
        data = ds.read(window=window)
        transform = ds.window_transform(window)
        return data, transform, ds.crs

def bounds_to_geom(minx: float, miny: float, maxx: float, maxy: float):
    _, _, _, box, *_ = _lazy_imports()
    if box is None:
        raise RuntimeError("shapely is unavailable in this environment.")
    return box(minx, miny, maxx, maxy)