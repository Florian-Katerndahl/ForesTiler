# ForesTiler — Create Image Tiles From Large Input Rasters According to a Classified Mask Vector File

ForesTiler provides a CLI tool to create image tiles from large input rasters according to a classified mask vector file.
The goal is to export images that are completely covered by a class polygon.
They can be saved either as PNGs to directly feed them into machine learning frameworks which may not interop with geodata or
as GeoTIFFs when retention of geographic information is deemed important.

## Installation

> :warning: Please note, that you need to have at least Python 3.9 installed.

You can install `forestiler` via pip by running

```bash
pip install forestiler
```

## Usage

> :warning: Only north-up images are supported. If your raster images are rotated, please transform them first!

```
usage: forestile [-h] [--no-progress] [--pad] [--kernel-size KERNEL_SIZE] [--stride STRIDE] 
                 --vector-mask VECTOR_MASK [--class-field CLASS_FIELD] [--all-classes]
                 [--classes CLASSES [CLASSES ...]] [--input-glob INPUT_GLOB] [--geo-tiff]
                 input out

forestile creates image tiles from large input rasters according to a classified mask vector file.

positional arguments:
  input                 Directory containing raster files to tile.
  out                   Directory where output files should be stored. 
                        May not exist prior to program invocation.

optional arguments:
  -h, --help            show this help message and exit
  --no-progress         Disable progress bar
  --pad                 Disable padding of input images.
  --kernel-size KERNEL_SIZE
                        Kernel size in pixels.
  --stride STRIDE       Stride of kernel.
  --vector-mask VECTOR_MASK
                        Path to vector file. Always reads first layer, 
                        if driver supports multi-layerr files (e.g. Geopackages).
  --class-field CLASS_FIELD
                        Attribute field containing class values.
  --all-classes         Generate image chips for all unique values in class field.
  --classes CLASSES [CLASSES ...]
                        List of classes to build image tiles for.
  --input-glob INPUT_GLOB
                        Optional glob pattern to filter files in input directory.
  --geo-tiff            Store image chips as GeoTiffs instead of PNGs.

Copyright: Florian Katerndahl <florian@katerndahl.com>
```

### Example Usage

Given a directory structure like the one listed below, the following command reads the file `mask.gpkg` and queries all geometries where the attribute "tree" matches "oak".
All output image chips, here as georeferenced TIFF files, are written into the directory `output` (which may not exist prior to program invocation).

```bash
forestile --stride 50 --vector-mask mask.gpkg --class-field tree --classes --geo-tiff "oak" rasters/ output/
```

```bash
.
├── output/
├── rasters/
│   ├── truedop20rgb_386_5808_2_be_2020.tif
│   ├── truedop20rgb_386_5810_2_be_2020.tif
│   ├── truedop20rgb_388_5808_2_be_2020.tif
│   └── truedop20rgb_388_5810_2_be_2020.tif
└── mask.gpkg
```

## Contribution

Bug reports, suggestions and feature requests are always welcomed. Please [open an issue](https://github.com/Florian-Katerndahl/ForesTiler/issues) on GitHub.

## Acknowledgements

This package was developed at the Remote Sensing Lab at Freie Universität Berlin.
