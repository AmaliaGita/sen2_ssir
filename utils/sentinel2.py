"""Sentinel-2 L2A discovery and GDAL-based band loading."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

import numpy as np

from .config import COARSE_BANDS_MODEL_ORDER, FINE_BANDS_MODEL_ORDER


PathLike = Union[str, Path]
PixelRoi = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SceneMetadata:
    scene_id: str
    source: str
    width: int
    height: int
    projection: str
    geotransform: Tuple[float, float, float, float, float, float]
    fine_band_descriptions: Dict[str, str]
    coarse_band_descriptions: Dict[str, str]


@dataclass
class Sentinel2Scene:
    fine: np.ndarray
    coarse: np.ndarray
    metadata: SceneMetadata


def discover_scenes(directory: PathLike) -> List[Path]:
    """Return Sentinel-2 products placed below a scene directory."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {directory}")

    safe_directories = sorted(
        path for path in directory.rglob("*.SAFE") if path.is_dir()
    )
    if safe_directories:
        return safe_directories

    metadata_files = sorted(directory.rglob("MTD_MSIL2A.xml"))
    if metadata_files:
        return metadata_files

    zip_files = sorted(directory.rglob("*.zip"))
    return zip_files


def require_single_scene(directory: PathLike) -> Path:
    """Find exactly one testing scene and provide an actionable error otherwise."""

    scenes = discover_scenes(directory)
    if len(scenes) != 1:
        raise RuntimeError(
            f"Expected exactly one Sentinel-2 L2A scene in {Path(directory)}, "
            f"but found {len(scenes)}."
        )
    return scenes[0]


def resolve_product_source(scene: PathLike) -> Union[Path, str]:
    """Resolve a SAFE directory to its L2A metadata file."""

    scene = Path(scene).expanduser().resolve()
    if scene.is_dir():
        metadata = scene / "MTD_MSIL2A.xml"
        if not metadata.is_file():
            matches = sorted(scene.glob("MTD_MSIL2A*.xml"))
            if not matches:
                raise FileNotFoundError(f"No MTD_MSIL2A.xml found in {scene}")
            metadata = matches[0]
        return metadata
    if scene.is_file() and scene.suffix.lower() == ".xml":
        return scene
    if scene.is_file() and scene.suffix.lower() == ".zip":
        with ZipFile(scene) as archive:
            metadata_members = sorted(
                name
                for name in archive.namelist()
                if name.upper().endswith("/MTD_MSIL2A.XML")
                or name.upper() == "MTD_MSIL2A.XML"
            )
        if not metadata_members:
            raise FileNotFoundError(f"No MTD_MSIL2A.xml found in {scene}")
        # GDAL's /vsizip/ virtual filesystem reads the product without extraction.
        return f"/vsizip/{scene.as_posix()}/{metadata_members[0]}"
    raise FileNotFoundError(f"Unsupported or missing Sentinel-2 product: {scene}")


def scene_identifier(scene: PathLike) -> str:
    scene = Path(scene)
    if scene.name.upper().endswith(".SAFE"):
        return scene.stem
    if scene.name.upper().startswith("MTD_MSIL2A"):
        return scene.parent.stem
    identifier = scene.stem
    if identifier.upper().endswith(".SAFE"):
        identifier = Path(identifier).stem
    return identifier


def _import_gdal():
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError(
            "GDAL is required for Sentinel-2 input. Activate the documented "
            "Conda environment before running the notebooks."
        ) from exc
    gdal.UseExceptions()
    return gdal


def _resolution_subdatasets(product_dataset):
    subdatasets = {10: [], 20: [], 60: []}
    for name, description in product_dataset.GetSubDatasets():
        for resolution in subdatasets:
            if f"{resolution}m resolution" in description.lower():
                subdatasets[resolution].append((name, description))
                break
    return subdatasets


def _utm_label(description: str) -> Optional[str]:
    match = re.search(r"UTM\s+\d+[NS]", description, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def _select_resolution_pair(subdatasets):
    if not subdatasets[10] or not subdatasets[20]:
        raise RuntimeError("The product does not expose both 10 m and 20 m subdatasets")

    selected_10 = subdatasets[10][0]
    selected_utm = _utm_label(selected_10[1])
    selected_20 = next(
        (
            candidate
            for candidate in subdatasets[20]
            if selected_utm and _utm_label(candidate[1]) == selected_utm
        ),
        subdatasets[20][0],
    )
    return selected_10, selected_20


def _short_band_name(description: str) -> str:
    token = description.split(",", 1)[0].split(" ", 1)[0].strip().upper()
    return token


def _band_lookup(dataset) -> Tuple[Dict[str, int], Dict[str, str]]:
    indices: Dict[str, int] = {}
    descriptions: Dict[str, str] = {}
    for index in range(1, dataset.RasterCount + 1):
        description = dataset.GetRasterBand(index).GetDescription()
        short_name = _short_band_name(description)
        indices[short_name] = index
        descriptions[short_name] = description
    return indices, descriptions


def _normalise_roi(width: int, height: int, roi: Optional[PixelRoi]) -> PixelRoi:
    if roi is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        if len(roi) != 4:
            raise ValueError("roi must be (x_min, y_min, x_max, y_max)")
        x0, y0, x1, y1 = (int(value) for value in roi)
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

    # Alignment to the native 20 m grid is required for paired reads.
    x0 -= x0 % 2
    y0 -= y0 % 2
    x1 -= x1 % 2
    y1 -= y1 % 2
    if x1 <= x0 or y1 <= y0:
        raise ValueError("roi is empty after alignment to the 20 m grid")
    return x0, y0, x1, y1


def _read_selected_bands(
    dataset,
    band_names: Sequence[str],
    xoff: int,
    yoff: int,
    width: int,
    height: int,
) -> Tuple[np.ndarray, Dict[str, str]]:
    indices, descriptions = _band_lookup(dataset)
    missing = [band for band in band_names if band not in indices]
    if missing:
        raise RuntimeError(f"Missing required Sentinel-2 bands: {missing}")

    arrays = []
    selected_descriptions = {}
    for band in band_names:
        array = dataset.GetRasterBand(indices[band]).ReadAsArray(
            xoff=xoff,
            yoff=yoff,
            win_xsize=width,
            win_ysize=height,
        )
        # Preserve GDAL's native integer type.
        # SciPy's Gaussian filter at that type before storing Float32 patches.
        arrays.append(np.asarray(array))
        selected_descriptions[band] = descriptions[band]
    return np.stack(arrays, axis=-1), selected_descriptions


def read_sentinel2_l2a(
    scene: PathLike,
    roi: Optional[PixelRoi] = None,
) -> Sentinel2Scene:
    """Read the ten bands used by Sen2-SSIR from one L2A product.

    ``roi`` uses exclusive 10 m pixel coordinates ``(x0, y0, x1, y1)``.
    Raw product values are intentionally retained to match legacy training.
    """

    gdal = _import_gdal()
    source = resolve_product_source(scene)
    product = gdal.Open(str(source))
    if product is None:
        raise RuntimeError(f"GDAL could not open Sentinel-2 product: {source}")

    subdatasets = _resolution_subdatasets(product)
    selected_10, selected_20 = _select_resolution_pair(subdatasets)
    dataset_10 = gdal.Open(selected_10[0])
    dataset_20 = gdal.Open(selected_20[0])
    if dataset_10 is None or dataset_20 is None:
        raise RuntimeError("GDAL could not open the selected Sentinel-2 subdatasets")

    x0, y0, x1, y1 = _normalise_roi(
        dataset_10.RasterXSize,
        dataset_10.RasterYSize,
        roi,
    )
    width = x1 - x0
    height = y1 - y0

    fine, fine_descriptions = _read_selected_bands(
        dataset_10,
        FINE_BANDS_MODEL_ORDER,
        x0,
        y0,
        width,
        height,
    )
    coarse, coarse_descriptions = _read_selected_bands(
        dataset_20,
        COARSE_BANDS_MODEL_ORDER,
        x0 // 2,
        y0 // 2,
        width // 2,
        height // 2,
    )

    geotransform = list(dataset_10.GetGeoTransform())
    geotransform[0] += x0 * geotransform[1] + y0 * geotransform[2]
    geotransform[3] += x0 * geotransform[4] + y0 * geotransform[5]
    metadata = SceneMetadata(
        scene_id=scene_identifier(scene),
        source=str(source),
        width=width,
        height=height,
        projection=dataset_10.GetProjection(),
        geotransform=tuple(geotransform),
        fine_band_descriptions=fine_descriptions,
        coarse_band_descriptions=coarse_descriptions,
    )
    return Sentinel2Scene(fine=fine, coarse=coarse, metadata=metadata)


def validate_scene_shapes(scene: Sentinel2Scene) -> None:
    """Validate band counts and the native 2x spatial relationship."""

    if scene.fine.ndim != 3 or scene.fine.shape[-1] != 4:
        raise ValueError("Fine input must contain four bands")
    if scene.coarse.ndim != 3 or scene.coarse.shape[-1] != 6:
        raise ValueError("Coarse input must contain six bands")
    if scene.fine.shape[0] != scene.coarse.shape[0] * 2:
        raise ValueError("Fine and coarse image heights do not have a 2x relationship")
    if scene.fine.shape[1] != scene.coarse.shape[1] * 2:
        raise ValueError("Fine and coarse image widths do not have a 2x relationship")
