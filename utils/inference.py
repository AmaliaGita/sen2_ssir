"""Simulated and real-image inference for Sen2-SSIR."""

from pathlib import Path
from typing import Optional

import numpy as np

from .config import (
    FINE_BANDS_MODEL_ORDER,
    InferenceConfig,
    MODEL_SCALE,
    OUTPUT_BANDS,
)
from .patches import count_inference_tiles, iter_inference_batches
from .sentinel2 import PixelRoi, Sentinel2Scene, read_sentinel2_l2a, validate_scene_shapes


def predict_tiled_to_npy(
    model,
    fine: np.ndarray,
    coarse: np.ndarray,
    output_path,
    config: InferenceConfig = InferenceConfig(),
    verbose: bool = False,
) -> Path:
    """Predict six high-resolution bands into a disk-backed NPY array."""

    config.validate()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(fine.shape[0], fine.shape[1], 6),
    )
    total_tiles = count_inference_tiles(fine.shape[:2], config.patch_size, config.border)
    report_interval = max(1, total_tiles // 20)
    next_report = report_interval
    completed_tiles = 0

    for batch in iter_inference_batches(
        fine,
        coarse,
        patch_size=config.patch_size,
        border=config.border,
        batch_size=config.batch_size,
    ):
        predictions = model.predict_on_batch([batch.fine, batch.coarse])
        predictions = np.asarray(predictions, dtype=np.float32) * MODEL_SCALE
        for prediction, position in zip(predictions, batch.positions):
            y0 = config.border
            x0 = config.border
            output[
                position.y : position.y + position.height,
                position.x : position.x + position.width,
            ] = prediction[
                y0 : y0 + position.height,
                x0 : x0 + position.width,
            ]
        completed_tiles += len(batch.positions)
        if verbose and (completed_tiles >= next_report or completed_tiles == total_tiles):
            print(f"Simulated inference: {completed_tiles}/{total_tiles} tiles")
            while next_report <= completed_tiles:
                next_report += report_interval
    output.flush()
    del output
    return output_path


def _geotiff_creation_options(config: InferenceConfig):
    compression = config.geotiff_compression.upper()
    options = [
        "TILED=YES",
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
        "BIGTIFF=IF_SAFER",
    ]
    if compression == "NONE":
        return options

    predictor = "3" if config.geotiff_dtype.lower() == "float32" else "2"
    options.extend(
        [
            f"COMPRESS={compression}",
            f"PREDICTOR={predictor}",
            "NUM_THREADS=ALL_CPUS",
        ]
    )
    if compression == "DEFLATE":
        options.append(f"ZLEVEL={config.geotiff_compression_level}")
    elif compression == "ZSTD":
        options.append(f"ZSTD_LEVEL={config.geotiff_compression_level}")
    return options


def _create_output_geotiff(
    scene: Sentinel2Scene,
    output_path: Path,
    config: InferenceConfig,
):
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError("GDAL is required to write the inference GeoTIFF") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    output_dtype = (
        gdal.GDT_Float32
        if config.geotiff_dtype.lower() == "float32"
        else gdal.GDT_UInt16
    )
    dataset = driver.Create(
        str(output_path),
        scene.metadata.width,
        scene.metadata.height,
        len(OUTPUT_BANDS),
        output_dtype,
        options=_geotiff_creation_options(config),
    )
    if dataset is None:
        raise RuntimeError(f"GDAL could not create output: {output_path}")
    dataset.SetGeoTransform(scene.metadata.geotransform)
    dataset.SetProjection(scene.metadata.projection)
    dataset.SetMetadataItem("MODEL", "Sen2-SSIR")
    dataset.SetMetadataItem("COMPRESSION", config.geotiff_compression.upper())
    dataset.SetMetadataItem("STORAGE_DTYPE", config.geotiff_dtype.lower())
    if config.geotiff_dtype.lower() == "float32":
        dataset.SetMetadataItem("UNITS", "surface reflectance")
        dataset.SetMetadataItem(
            "SCALE_APPLIED",
            f"input digital number divided by {config.output_scale:g}",
        )
    else:
        dataset.SetMetadataItem("UNITS", "Sentinel-2 L2A digital number")
        dataset.SetMetadataItem(
            "SCALE_FACTOR",
            f"{1.0 / config.output_scale:.12g}",
        )
    for output_index, band_name in enumerate(OUTPUT_BANDS, start=1):
        band = dataset.GetRasterBand(output_index)
        band.SetDescription(band_name)
        band.SetMetadataItem("WAVELENGTH_BAND_NAME", band_name)
        if config.geotiff_dtype.lower() == "uint16":
            band.SetScale(1.0 / config.output_scale)
            band.SetOffset(0.0)
    return dataset


def _encode_geotiff_values(values: np.ndarray, config: InferenceConfig) -> np.ndarray:
    if config.geotiff_dtype.lower() == "float32":
        return values.astype(np.float32, copy=False) / np.float32(config.output_scale)
    return np.clip(np.rint(values), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _write_native_fine_bands(
    dataset,
    fine: np.ndarray,
    config: InferenceConfig,
) -> None:
    output_indices = {band: index + 1 for index, band in enumerate(OUTPUT_BANDS)}
    for source_index, band_name in enumerate(FINE_BANDS_MODEL_ORDER):
        dataset.GetRasterBand(output_indices[band_name]).WriteArray(
            _encode_geotiff_values(fine[..., source_index], config)
        )


def superresolve_scene_to_geotiff(
    model,
    scene_path,
    output_path,
    config: InferenceConfig = InferenceConfig(),
    roi: Optional[PixelRoi] = None,
    verbose: bool = False,
) -> Path:
    """Create a georeferenced 10 m, ten-band Sentinel-2 product.

    Native B2, B3, B4, and B8 are copied to the output. Sen2-SSIR predicts
    B5, B6, B7, B8A, B11, and B12 at the same 10 m grid. Values in the final
    GeoTIFF storage is controlled by ``config``. Float32 output contains
    surface reflectance (digital number divided by ``output_scale``). Compact
    UInt16 output stores rounded digital numbers and records the reciprocal
    scale factor in each band.
    """

    config.validate()
    scene = read_sentinel2_l2a(scene_path, roi=roi)
    validate_scene_shapes(scene)
    output_path = Path(output_path).expanduser().resolve()
    dataset = _create_output_geotiff(scene, output_path, config)
    try:
        _write_native_fine_bands(dataset, scene.fine, config)
        predicted_band_indices = {
            band_name: OUTPUT_BANDS.index(band_name) + 1
            for band_name in ("B5", "B6", "B7", "B8A", "B11", "B12")
        }
        total_tiles = count_inference_tiles(
            scene.fine.shape[:2],
            config.patch_size,
            config.border,
        )
        report_interval = max(1, total_tiles // 20)
        next_report = report_interval
        completed_tiles = 0

        for batch in iter_inference_batches(
            scene.fine,
            scene.coarse,
            patch_size=config.patch_size,
            border=config.border,
            batch_size=config.batch_size,
        ):
            predictions = model.predict_on_batch([batch.fine, batch.coarse])
            predictions = np.asarray(predictions, dtype=np.float32)
            predictions *= MODEL_SCALE
            predictions = _encode_geotiff_values(predictions, config)
            for prediction, position in zip(predictions, batch.positions):
                cropped = prediction[
                    config.border : config.border + position.height,
                    config.border : config.border + position.width,
                ]
                for channel, band_name in enumerate(
                    ("B5", "B6", "B7", "B8A", "B11", "B12")
                ):
                    dataset.GetRasterBand(predicted_band_indices[band_name]).WriteArray(
                        cropped[..., channel],
                        xoff=position.x,
                        yoff=position.y,
                    )
            completed_tiles += len(batch.positions)
            if verbose and (
                completed_tiles >= next_report or completed_tiles == total_tiles
            ):
                print(f"Real inference: {completed_tiles}/{total_tiles} tiles")
                while next_report <= completed_tiles:
                    next_report += report_interval
        dataset.FlushCache()
    finally:
        dataset = None
    return output_path
