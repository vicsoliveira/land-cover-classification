import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rasterio.enums import Resampling

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BANDS = {
    "B02": os.path.join(BASE_DIR, "data_raw", "B02.tif"),
    "B03": os.path.join(BASE_DIR, "data_raw", "B03.tif"),
    "B04": os.path.join(BASE_DIR, "data_raw", "B04.tif"),
    "B08": os.path.join(BASE_DIR, "data_raw", "B08.tif"),
}

OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
OUTPUT_RASTER = os.path.join(OUTPUT_DIR, "classified.tif")
OUTPUT_PNG    = os.path.join(OUTPUT_DIR, "classified.png")
OUTPUT_STATS  = os.path.join(OUTPUT_DIR, "stats.csv")

# ==============================
# THRESHOLDS — tweak these to tune results
# ==============================

NDVI_DENSE_VEG  = 0.30   # lowered from 0.35 — more pixels become dense vegetation
NDVI_SPARSE_VEG = 0.16   # unchanged
NDWI_WATER      = 0.10   # lowered from 0.15 — captures more water bodies
NDWI_URBAN_MAX  = -0.25  # unchanged
NDVI_URBAN_MAX  = 0.10   # unchanged

CLASS_NAMES = {
    0: "Unclassified",
    1: "Dense Vegetation",
    2: "Sparse Vegetation",
    3: "Water",
    4: "Urban",
    5: "Bare Soil",
}

CLASS_COLORS = {
    0: [0.2, 0.2, 0.2],   # Unclassified — dark gray
    1: [0.0, 0.4, 0.0],   # Dense Vegetation — dark green
    2: [0.6, 0.8, 0.2],   # Sparse Vegetation — light green
    3: [0.0, 0.4, 0.8],   # Water — blue
    4: [0.8, 0.2, 0.2],   # Urban — red
    5: [0.8, 0.7, 0.4],   # Bare Soil — tan
}

# ==============================
# FUNCTIONS
# ==============================

def load_and_stack_bands(band_paths):
    band_arrays  = []
    base_profile = None
    base_shape   = None

    for i, (band_name, path) in enumerate(band_paths.items()):
        print(f"Loading {band_name}: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with rasterio.open(path) as src:
            if i == 0:
                base_profile = src.profile
                base_shape   = (src.height, src.width)
                band         = src.read(1)
            else:
                band = src.read(
                    1,
                    out_shape=base_shape,
                    resampling=Resampling.bilinear
                )
        band_arrays.append(band)

    return np.stack(band_arrays), base_profile


def calculate_indices(bands, band_names):
    blue  = bands[band_names.index("B02")].astype(float)
    green = bands[band_names.index("B03")].astype(float)
    red   = bands[band_names.index("B04")].astype(float)
    nir   = bands[band_names.index("B08")].astype(float)

    # NDVI — vegetation index
    ndvi = (nir - red)   / (nir + red   + 1e-10)

    # NDWI — water index (also suppresses urban)
    ndwi = (green - nir) / (green + nir + 1e-10)

    # BRI — Blue Reflectance Index (urban surfaces reflect more blue than bare soil)
    bri  = (blue - red)  / (blue + red  + 1e-10)

    return ndvi, ndwi, bri


def classify(ndvi, ndwi, bri):
    """
    Rule-based classification.
    Priority order (last written = highest priority):
    Bare Soil < Urban < Sparse Veg < Dense Veg < Water
    """
    classified = np.zeros(ndvi.shape, dtype=np.uint8)

    # Bare Soil — low NDVI, not water
    classified[
        (ndvi < NDVI_SPARSE_VEG) &
        (ndwi < NDWI_WATER)
    ] = 5

    # Urban — strict: very low NDVI + strongly negative NDWI + positive BRI
    classified[
        (ndvi < NDVI_URBAN_MAX) &
        (ndwi < NDWI_URBAN_MAX) &
        (bri  > 0)
    ] = 4

    # Sparse Vegetation
    classified[
        (ndvi >= NDVI_SPARSE_VEG) &
        (ndvi <  NDVI_DENSE_VEG)
    ] = 2

    # Dense Vegetation
    classified[
        ndvi >= NDVI_DENSE_VEG
    ] = 1

    # Water — highest priority, overwrites everything
    classified[
        ndwi >= NDWI_WATER
    ] = 3

    return classified


def calculate_stats(classified):
    unique, counts = np.unique(classified, return_counts=True)
    total = counts.sum()
    data  = []
    for u, c in zip(unique, counts):
        data.append({
            "class_id":    int(u),
            "class_name":  CLASS_NAMES.get(int(u), "Unknown"),
            "pixel_count": int(c),
            "percentage":  round((c / total) * 100, 2)
        })
    return pd.DataFrame(data)


def save_raster(output_path, classified, profile):
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classified, 1)


def save_png(classified, path):
    h, w = classified.shape
    rgb  = np.zeros((h, w, 3), dtype=float)

    for class_id, color in CLASS_COLORS.items():
        mask = classified == class_id
        for c in range(3):
            rgb[mask, c] = color[c]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)

    legend_elements = [
        Patch(facecolor=CLASS_COLORS[k], label=CLASS_NAMES[k])
        for k in CLASS_NAMES
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.set_title("Land Use Classification (Rule-Based)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"PNG saved: {path}")


def print_index_stats(ndvi, ndwi, bri):
    print("\nSpectral index ranges:")
    print(f"  NDVI: {ndvi.min():.3f} to {ndvi.max():.3f}  |  mean: {ndvi.mean():.3f}")
    print(f"  NDWI: {ndwi.min():.3f} to {ndwi.max():.3f}  |  mean: {ndwi.mean():.3f}")
    print(f"  BRI:  {bri.min():.3f}  to {bri.max():.3f}   |  mean: {bri.mean():.3f}")
    print()


# ==============================
# MAIN
# ==============================

def main():
    print("\n=== STARTING PROCESS ===\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load bands
    bands, profile = load_and_stack_bands(BANDS)
    band_names = list(BANDS.keys())

    # 2. Calculate indices
    print("Calculating spectral indices...")
    ndvi, ndwi, bri = calculate_indices(bands, band_names)
    print_index_stats(ndvi, ndwi, bri)

    # 3. Classify
    print("Classifying...")
    classified = classify(ndvi, ndwi, bri)

    # 4. Save outputs
    print("Saving outputs...")
    save_raster(OUTPUT_RASTER, classified, profile)
    save_png(classified, OUTPUT_PNG)

    stats = calculate_stats(classified)
    stats.to_csv(OUTPUT_STATS, index=False)

    print("\n=== DONE ===\n")
    print(stats.to_string(index=False))


# ==============================

if __name__ == "__main__":
    main()