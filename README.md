# Sen2-SSIR
This source code is official implementation for "An Inception-Residual Sentinel-2 Single-Image Super-Resolution Model for Enhancing Ecological Analysis in Taiwan".

Sen2-SSIR super-resolves the six native 20 m Sentinel-2 L2A bands (B5, B6,
B7, B8A, B11, and B12) to the native 10 m grid. It combines those predictions
with B2, B3, B4, and B8 to produce one georeferenced ten-band 10 m GeoTIFF.

## Repository layout

```text
├── 1_Train_Sen2-SSIR.ipynb
├── 2_Evaluate_and_Infer_Sen2-SSIR.ipynb
├── Data/
│   ├── Training_Scene/   # place one or more L2A products here
│   ├── Testing_Scene/    # place exactly one L2A product here
│   └── Prepared_Data/    # generated training patches
├── Results/              # generated checkpoints, metrics, arrays, GeoTIFFs
├── utils/                # all Python source code
└── weights/
    └── Sen2-SSIR_pretrained.weights.h5
```

Input products may be an unpacked `.SAFE` directory, `MTD_MSIL2A.xml`, or a
Sentinel-2 L2A `.zip` that GDAL can open. Do not place multiple products in
`Data/Testing_Scene`: Notebook 2 deliberately requires exactly one scene so
results cannot silently be associated with the wrong product.

## Environment

The implementation was verified with Python 3.8, TensorFlow/Keras 2.10 GPU, NumPy 1.24.4, SciPy 1.10.1,
scikit-image 0.21, h5py 3.9, Matplotlib 3.7.3, and GDAL 3.4.3.

## 1. Training

Place one or more L2A products below `Data/Training_Scene`, then run
`1_Train_Sen2-SSIR.ipynb`. Select one of three modes in its configuration cell:

- `scratch`: initialize Sen2-SSIR randomly and train from the beginning.
- `pretrained`: load the published weights and fine-tune on the new scenes.
- `resume`: restore `Results/Training/Sen2-SSIR_latest.h5`, including optimizer
  state, and continue an interrupted run.


## 2. Evaluation and real inference

Place exactly one L2A product below `Data/Testing_Scene`, then run
`2_Evaluate_and_Infer_Sen2-SSIR.ipynb`. It performs both workflows:

1. **Simulated assessment:** native 10 m bands are degraded to 20 m; native
   20 m bands are degraded to 40 m; the native 20 m bands are the reference.
   The three manuscript metrics (RMSE, SRE, and ERGAS) are written as JSON as
   aggregate six-band results and separately for B5, B6, B7, B8A, B11, and B12.
2. **Real inference:** native 10 m and 20 m inputs are used to create a ten-band
   10 m GeoTIFF in canonical order
   `B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12`.

## 3. Sentinel-2 Data Used in the Original Paper
The following table summarizes the Sentinel-2 tile codes, sensing dates, and their corresponding roles in model training and independent testing.

| Tile Code | Sensing Date | Used For |
|-----------|--------------|----------|
| T51RUH | 2023/03/06 | Training |
| T51QTG | 2023/03/06 | Training |
| T51QUG | 2023/03/01 | Training |
| T51QTF | 2023/03/01 | Original Independent Testing |
| T51QUV | 2023/03/23 | Additional Independent Testing |
| T52SCE | 2024/04/01 | Additional Independent Testing |
| T52SFB | 2024/05/10 | Additional Independent Testing |
| T51QTF | 2024/12/05 | Additional Independent Testing |
| T52SCE | 2025/04/26 | Additional Independent Testing |

## Acknowledgments
This source code has benefited from the work and implementation of the [DSen2 repository](https://github.com/lanha/DSen2).