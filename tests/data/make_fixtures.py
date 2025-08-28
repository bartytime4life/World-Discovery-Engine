# tests/data/make_fixtures.py
import json, math, os, struct
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.crs import CRS

ROOT = Path(__file__).parent
AOI = ROOT / "aoi"
RAST = ROOT / "rasters"
TS = ROOT / "time_series"
TXT = ROOT / "text"
GR = ROOT / "graphs"
for p in (AOI, RAST, TS, TXT, GR): p.mkdir(parents=True, exist_ok=True)

def write_geojson_bbox(fp):
    # ~0.05° box centered near (-3.45, -60.45)
    min_lat, min_lon = -3.50, -60.50
    max_lat, max_lon = -3.45, -60.45
    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "test_bbox"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],[max_lon, min_lat],
                    [max_lon, max_lat],[min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            }
        }]
    }
    fp.write_text(json.dumps(gj, indent=2))

def write_geotiff(path, arr, dtype, transform, crs=CRS.from_epsg(4326)):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[-2],
        "width": arr.shape[-1],
        "count": 1 if arr.ndim == 2 else arr.shape[0],
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": False,
        "compress": "LZW"
    }
    with rasterio.open(path, "w", **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr.astype(dtype), 1)
        else:
            for b in range(arr.shape[0]):
                dst.write(arr[b].astype(dtype), b + 1)

def make_optical_rgb():
    H=W=256
    transform = from_origin(-60.50, -3.45, 0.0002, 0.0002)  # ~22m px at equator
    rng = np.random.default_rng(42)
    base = rng.integers(60, 90, size=(3, H, W), dtype=np.uint8)
    # Inject bright circular anomaly at center
    cy, cx, r = H//2, W//2, 20
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy)**2 + (xx - cx)**2 <= r*r
    base[:, mask] = 255
    write_geotiff(RAST/"optical_rgb_256.tif", base, "uint8", transform)

def make_dem():
    H=W=50
    transform = from_origin(-60.50, -3.45, 0.001, 0.001)  # coarse ~111m px
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H/2, W/2
    sigma = 7.5
    mound = 5.0 * np.exp(-(((x-cx)**2 + (y-cy)**2)/(2*sigma**2)))
    base = 50.0 + mound + 0.2*np.sin(x/3)  # gentle relief + mound
    write_geotiff(RAST/"dem_50m.tif", base.astype("float32"), "float32", transform)

def make_soil_p():
    H=W=50
    transform = from_origin(-60.50, -3.45, 0.001, 0.001)
    arr = np.full((H,W), 200, dtype=np.uint16)  # ppm baseline
    arr[H//2, W//2] = 1200  # P spike beneath mound
    write_geotiff(RAST/"soil_p_ppm_50m.tif", arr, "uint16", transform)

def make_landcover():
    H=W=50
    transform = from_origin(-60.50, -3.45, 0.001, 0.001)
    lc = np.ones((H,W), dtype=np.uint8)  # 1=forest
    lc[18:32,18:32] = 2  # 2=palm/indicator species patch over site
    write_geotiff(RAST/"landcover_50m.tif", lc, "uint8", transform)

def make_sar():
    H=W=50
    transform = from_origin(-60.50, -3.45, 0.001, 0.001)
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H/2, W/2
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    arr = -10 + 3*np.exp(-(r-8)**2/10)  # dB-ish ring enhancement
    write_geotiff(RAST/"sar_backscatter_50m.tif", arr.astype("float32"), "float32", transform)

def write_ndvi_csv():
    p = TS/"ndvi_site_vs_bg.csv"
    # months 1..12, site stays ~0.8; background dips in 7–9
    site = [0.75,0.78,0.80,0.82,0.80,0.78,0.79,0.81,0.83,0.80,0.78,0.76]
    bg   = [0.75,0.76,0.78,0.80,0.78,0.75,0.60,0.55,0.58,0.70,0.72,0.74]
    with p.open("w") as f:
        f.write("month,site_ndvi,bg_ndvi\n")
        for m,(s,b) in enumerate(zip(site,bg), start=1):
            f.write(f"{m},{s},{b}\n")

def write_snippet():
    TXT.joinpath("historical_snippet.txt").write_text(
        "…to the east bend of the great river we found black fertile soils "
        "and old mounds; the inhabitants spoke of ancient gardens…\n"
    )

def write_pag_gml():
    GR.joinpath("pag_mock.gml").write_text(
        'graph [\n'
        '  directed 1\n'
        '  node [ id 0 label "elevation" ]\n'
        '  node [ id 1 label "soil" ]\n'
        '  node [ id 2 label "ndvi" ]\n'
        '  edge [ source 0 target 1 ]\n'
        '  edge [ source 1 target 2 ]\n'
        ']\n'
    )

if __name__ == "__main__":
    write_geojson_bbox(AOI/"test_bbox.geojson")
    make_optical_rgb()
    make_dem()
    make_soil_p()
    make_landcover()
    make_sar()
    write_ndvi_csv()
    write_snippet()
    write_pag_gml()
    print("✓ fixtures created in tests/data/")
