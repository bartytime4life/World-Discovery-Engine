# 📦 `tests/data/` — offline test fixtures

## Structure

```text
tests/
  data/
    README.md
    aoi/
      test_bbox.geojson
    rasters/
      optical_rgb_256.tif          # tiny 256×256 3-band (with injected anomaly)
      dem_50m.tif                  # 50×50 DEM with circular “mound”
      soil_p_ppm_50m.tif           # 50×50 soil phosphorus mock (high pixel)
      landcover_50m.tif            # 50×50 categorical floristics mock
      sar_backscatter_50m.tif      # 50×50 SAR intensity mock
    time_series/
      ndvi_site_vs_bg.csv          # 12-month NDVI series (site vs background)
    text/
      historical_snippet.txt       # OCR-like snippet (dummy)
    graphs/
      pag_mock.gml                 # elevation → soil → ndvi
    make_fixtures.py               # generator (idempotent)
```

---

## What’s in each file?

* **`aoi/test_bbox.geojson`** – 0.05° box; small enough for fast integration tests.
* **`rasters/optical_rgb_256.tif`** – 3-band uint8 “Sentinel-ish” tile (synthetic); a white disk (R=G=B=255) is injected at center to act as a bright anomaly.
* **`rasters/dem_50m.tif`** – float32 DEM with a smooth Gaussian bump (mound) centered in the scene.
* **`rasters/soil_p_ppm_50m.tif`** – uint16 soil phosphorus proxy with a single high-value pixel under the mound.
* **`rasters/landcover_50m.tif`** – uint8 categorical map (e.g., 1=forest, 2=palm/indicator species patch inside the AOI).
* **`rasters/sar_backscatter_50m.tif`** – float32 SAR intensity (log-scaled) with a subtle concentric pattern around the anomaly.
* **`time_series/ndvi_site_vs_bg.csv`** – 12-month NDVI (site stays \~0.8; background dips to \~0.55–0.60 in dry season months 7–9).
* **`text/historical_snippet.txt`** – a short, OCR-ish snippet mentioning “dark earth” near a bend in the river (for keyword tests).
* **`graphs/pag_mock.gml`** – small PAG with directed edges `elevation → soil → ndvi`.

---

## Generator script (`tests/data/make_fixtures.py`)

> Run once (or anytime) to (re)create fixtures deterministically.

```python
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
```

---

## Minimal `tests/data/README.md`

````markdown
# tests/data — Offline fixtures

Run:

```bash
python tests/data/make_fixtures.py
````

This creates tiny synthetic rasters (GeoTIFF), AOI, NDVI CSV, a mock historical snippet, and a small PAG graph.

Notes:

* All coordinates use EPSG:4326 (WGS84) and tiny extents for fast tests.
* Files are deterministic; re-running regenerates identical data.
* Keep file sizes small (<200 KB each) to keep the repo lightweight.

````

---

## Example pytest fixtures

Add to `tests/conftest.py` (or a dedicated `tests/fixtures.py`):

```python
import json
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def data_root():
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures(data_root):
    # Generate on demand
    import subprocess, sys
    script = data_root / "make_fixtures.py"
    subprocess.run([sys.executable, str(script)], check=True)

@pytest.fixture
def aoi_geojson(data_root):
    return json.loads((data_root / "aoi" / "test_bbox.geojson").read_text())

@pytest.fixture
def ndvi_csv_path(data_root):
    return data_root / "time_series" / "ndvi_site_vs_bg.csv"

@pytest.fixture
def rasters(data_root):
    r = data_root / "rasters"
    return {
        "optical": r / "optical_rgb_256.tif",
        "dem": r / "dem_50m.tif",
        "soil_p": r / "soil_p_ppm_50m.tif",
        "landcover": r / "landcover_50m.tif",
        "sar": r / "sar_backscatter_50m.tif",
    }
````

---

## How tests should consume these

* **Unit**: point functions at these tiny inputs (e.g., `detect` reads `optical_rgb_256.tif` and must find the bright disk).
* **Integration**: config points AOI → these rasters; pipeline must emit `candidates.json`, `reports/`, `pag/`, etc.
* **CI**: runs `make_fixtures.py` automatically (fast), no network access required.

---
