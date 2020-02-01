# Neat-EO.pink tools documentation
## neo compare
```
usage: neo compare [-h] [--mode {side,stack,list}] [--labels LABELS]
                   [--masks MASKS] [--config CONFIG]
                   [--images IMAGES [IMAGES ...]] [--cover COVER]
                   [--workers WORKERS] [--min MIN MIN MIN] [--max MAX MAX MAX]
                   [--vertical] [--geojson] [--format FORMAT] [--out OUT]
                   [--web_ui_base_url WEB_UI_BASE_URL]
                   [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 --mode {side,stack,list}           compare mode [default: side]
 --labels LABELS                    path to tiles labels directory [required for metrics filtering]
 --masks MASKS                      path to tiles masks directory [required for metrics filtering)
 --config CONFIG                    path to config file [required for metrics filtering, if no global config setting]
 --images IMAGES [IMAGES ...]       path to images directories [required for stack or side modes]
 --cover COVER                      path to csv tiles cover file, to filter tiles to tile [optional]
 --workers WORKERS                  number of workers [default: CPU]

Metrics Filtering:
 --min MIN MIN MIN                  skip tile if class below metric value [0-1] (e.g --min Building QoD 0.7)
 --max MAX MAX MAX                  skip tile if class above metric value [0-1] (e.g --max Building IoU 0.85)

Outputs:
 --vertical                         output vertical image aggregate [optionnal for side mode]
 --geojson                          output results as GeoJSON [optionnal for list mode]
 --format FORMAT                    output images file format [default: webp]
 --out OUT                          output path

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## neo cover
```
usage: neo cover [-h] [--dir DIR] [--bbox BBOX]
                 [--geojson GEOJSON [GEOJSON ...]] [--cover COVER]
                 [--raster RASTER [RASTER ...]] [--sql SQL] [--pg PG]
                 [--no_xyz] [--zoom ZOOM] [--type {cover,extent,geojson}]
                 [--union] [--splits SPLITS] [--out [OUT [OUT ...]]]

optional arguments:
 -h, --help                       show this help message and exit

Input [one among the following is required]:
 --dir DIR                        plain tiles dir path
 --bbox BBOX                      a lat/lon bbox: xmin,ymin,xmax,ymax or a bbox: xmin,xmin,xmax,xmax,EPSG:xxxx
 --geojson GEOJSON [GEOJSON ...]  path to GeoJSON features files
 --cover COVER                    a cover file path
 --raster RASTER [RASTER ...]     a raster file path
 --sql SQL                        SQL to retrieve geometry features (e.g SELECT geom FROM a_table)

Spatial DataBase [required with --sql input]:
 --pg PG                          PostgreSQL dsn using psycopg2 syntax (e.g 'dbname=db user=postgres')

Tiles:
 --no_xyz                         if set, tiles are not expected to be XYZ based.

Outputs:
 --zoom ZOOM                      zoom level of tiles [required, except with --dir or --cover inputs]
 --type {cover,extent,geojson}    Output type (default: cover)
 --union                          if set, union adjacent tiles, imply --type geojson
 --splits SPLITS                  if set, shuffle and split in several cover subpieces (e.g 50/15/35)
 --out [OUT [OUT ...]]            cover output paths [required except with --type extent]
```
## neo dataset
```
usage: neo dataset [-h] [--config CONFIG] --dataset DATASET [--cover COVER]
                   [--workers WORKERS] [--mode {check,weights}]

optional arguments:
 -h, --help              show this help message and exit
 --config CONFIG         path to config file [required, if no global config setting]
 --dataset DATASET       dataset path [required]
 --cover COVER           path to csv tiles cover file, to filter tiles dataset on [optional]
 --workers WORKERS       number of workers [default: CPU]
 --mode {check,weights}  dataset mode [default: check]
```
## neo download
```
usage: neo download [-h] --url URL [--type {XYZ,WMS}] [--rate RATE]
                    [--timeout TIMEOUT] [--workers WORKERS] --cover COVER
                    [--format FORMAT] --out OUT
                    [--web_ui_base_url WEB_UI_BASE_URL]
                    [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]

optional arguments:
 -h, --help                         show this help message and exit

Web Server:
 --url URL                          URL server endpoint, with: {z}/{x}/{y} or {xmin},{ymin},{xmax},{ymax} [required]
 --type {XYZ,WMS}                   service type [default: XYZ]
 --rate RATE                        download rate limit in max requests/seconds [default: 10]
 --timeout TIMEOUT                  download request timeout (in seconds) [default: 10]
 --workers WORKERS                  number of workers [default: same as --rate value]

Coverage to download:
 --cover COVER                      path to .csv tiles list [required]

Output:
 --format FORMAT                    file format to save images in [default: webp]
 --out OUT                          output directory path [required]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## neo export
```
usage: neo export [-h] --checkpoint CHECKPOINT [--type {onnx,jit,pth}]
                  [--nn NN] [--loader LOADER] [--doc_string DOC_STRING]
                  [--shape_in SHAPE_IN] [--shape_out SHAPE_OUT]
                  [--encoder ENCODER] --out OUT

optional arguments:
 -h, --help               show this help message and exit

Inputs:
 --checkpoint CHECKPOINT  model checkpoint to load [required]
 --type {onnx,jit,pth}    output type [default: onnx]

To set or override metadata pth parameters::
 --nn NN                  nn name
 --loader LOADER          nn loader
 --doc_string DOC_STRING  nn documentation abstract
 --shape_in SHAPE_IN      nn shape in (e.g 3,512,512)
 --shape_out SHAPE_OUT    nn shape_out  (e.g 2,512,512)
 --encoder ENCODER        nn encoder  (e.g resnet50)

Output:
 --out OUT                path to save export model to [required]
```
## neo extract
```
usage: neo extract [-h] --type TYPE --pbf PBF --out OUT

optional arguments:
 -h, --help   show this help message and exit

Inputs:
 --type TYPE  OSM feature type to extract (e.g Building, Road) [required]
 --pbf PBF    path to .osm.pbf file [required]

Output:
 --out OUT    GeoJSON output file path [required]
```
## neo info
```
usage: neo info [-h] [--version] [--processes] [--checkpoint CHECKPOINT]

optional arguments:
 -h, --help               show this help message and exit
 --version                if set, output Neat-EO.pink version only
 --processes              if set, output GPU processes list
 --checkpoint CHECKPOINT  if set with a .pth path, output related model metadata

Usages:
To kill GPU processes: neo info --processes | xargs sudo kill -9
```
## neo predict
```
usage: neo predict [-h] [--dataset DATASET] --checkpoint CHECKPOINT
                   [--config CONFIG] [--cover COVER] --out OUT [--metatiles]
                   [--bs BS] [--workers WORKERS]
                   [--web_ui_base_url WEB_UI_BASE_URL]
                   [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 --dataset DATASET                  predict dataset directory path [required]
 --checkpoint CHECKPOINT            path to the trained model to use [required]
 --config CONFIG                    path to config file [required, if no global config setting]
 --cover COVER                      path to csv tiles cover file, to filter tiles to predict [optional]

Outputs:
 --out OUT                          output directory path [required]
 --metatiles                        if set, use surrounding tiles to avoid margin effects

Performances:
 --bs BS                            batch size [default: CPU/GPU]
 --workers WORKERS                  number of pre-processing images workers, per GPU [default: batch_size]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## neo rasterize
```
usage: neo rasterize [-h] --cover COVER [--config CONFIG] --type TYPE
                     [--geojson GEOJSON [GEOJSON ...]] [--sql SQL] [--pg PG]
                     --out OUT [--append] [--ts TS]
                     [--web_ui_base_url WEB_UI_BASE_URL]
                     [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]

optional arguments:
 -h, --help                         show this help message and exit

Inputs [either --sql or --geojson is required]:
 --cover COVER                      path to csv tiles cover file [required]
 --config CONFIG                    path to config file [required, if no global config setting]
 --type TYPE                        type of features to rasterize (i.e class title) [required]
 --geojson GEOJSON [GEOJSON ...]    path to GeoJSON features files
 --sql SQL                          SQL to retrieve geometry features [e.g SELECT geom FROM table WHERE ST_Intersects(TILE_GEOM, geom)]
 --pg PG                            If set, override config PostgreSQL dsn.

Outputs:
 --out OUT                          output directory path [required]
 --append                           Append to existing tile if any, useful to multiclass labels
 --ts TS                            output tile size [default: 512,512]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## neo sat
```
usage: neo sat [-h] [--config CONFIG] [--pg PG] [--cover COVER]
               [--granules GRANULES [GRANULES ...]] [--scenes SCENES]
               [--level {2A,3A}] [--start START] [--end END] [--clouds CLOUDS]
               [--limit LIMIT] [--download] [--workers WORKERS]
               [--timeout TIMEOUT] [--out [OUT]]

optional arguments:
 -h, --help                          show this help message and exit
 --config CONFIG                     path to config file [required]
 --pg PG                             If set, override config PostgreSQL dsn.
 --out [OUT]                         output directory path [required if download is set]

Spatial extent [one among the following is required]:
 --cover COVER                       path to csv tiles cover file
 --granules GRANULES [GRANULES ...]  Military Grid Granules, (e.g 31TFL)
 --scenes SCENES                     Path to a Scenes UUID file

Filters:
 --level {2A,3A}                     Processing Level
 --start START                       YYYY-MM-DD starting date
 --end END                           YYYY-MM-DD end date
 --clouds CLOUDS                     max threshold for cloud coverage [0-100]
 --limit LIMIT                       max number of results per granule

Download:
 --download                          if set, perform also download operation.
 --workers WORKERS                   number of workers [default: 4]
 --timeout TIMEOUT                   download request timeout (in seconds) [default: 180]
```
## neo subset
```
usage: neo subset [-h] --dir DIR --cover COVER [--copy] [--delete] [--quiet]
                  [--out [OUT]] [--web_ui_base_url WEB_UI_BASE_URL]
                  [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 --dir DIR                          to XYZ tiles input dir path [required]
 --cover COVER                      path to csv cover file to filter dir by [required]

Alternate modes, as default is to create relative symlinks:
 --copy                             copy tiles from input to output
 --delete                           delete tiles listed in cover

Output:
 --quiet                            if set, suppress warning output
 --out [OUT]                        output dir path [required for copy]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## neo tile
```
usage: neo tile [-h] --rasters RASTERS [RASTERS ...] [--cover COVER] --zoom
                ZOOM [--ts TS] [--nodata [0-255]] [--nodata_threshold [0-100]]
                [--keep_borders] [--format FORMAT] --out OUT [--label]
                [--config CONFIG] [--workers WORKERS]
                [--web_ui_base_url WEB_UI_BASE_URL]
                [--web_ui_template WEB_UI_TEMPLATE] [--no_web_ui]

optional arguments:
 -h, --help                         show this help message and exit

Inputs:
 --rasters RASTERS [RASTERS ...]    path to raster files to tile [required]
 --cover COVER                      path to csv tiles cover file, to filter tiles to tile [optional]

Output:
 --zoom ZOOM                        zoom level of tiles [required]
 --ts TS                            tile size in pixels [default: 512,512]
 --nodata [0-255]                   nodata pixel value, used by default to remove coverage border's tile [default: 0]
 --nodata_threshold [0-100]         Skip tile if nodata pixel ratio > threshold. [default: 100]
 --keep_borders                     keep tiles even if borders are empty (nodata)
 --format FORMAT                    file format to save images in (e.g jpeg)
 --out OUT                          output directory path [required]

Labels:
 --label                            if set, generate label tiles
 --config CONFIG                    path to config file [required with --label, if no global config setting]

Performances:
 --workers WORKERS                  number of workers [default: raster files]

Web UI:
 --web_ui_base_url WEB_UI_BASE_URL  alternate Web UI base URL
 --web_ui_template WEB_UI_TEMPLATE  alternate Web UI template path
 --no_web_ui                        desactivate Web UI output
```
## neo train
```
usage: neo train [-h] [--config CONFIG] [--train_dataset TRAIN_DATASET]
                 [--eval_dataset EVAL_DATASET] [--cover COVER]
                 [--classes_weights CLASSES_WEIGHTS]
                 [--tiles_weights TILES_WEIGHTS] [--loader LOADER] [--bs BS]
                 [--lr LR] [--ts TS] [--nn NN] [--encoder ENCODER]
                 [--optimizer OPTIMIZER] [--loss LOSS] [--epochs EPOCHS]
                 [--resume] [--checkpoint CHECKPOINT] [--workers WORKERS]
                 [--saving SAVING] --out OUT

optional arguments:
 -h, --help                         show this help message and exit
 --config CONFIG                    path to config file [required, if no global config setting]

Dataset:
 --train_dataset TRAIN_DATASET      train dataset path [needed for train]
 --eval_dataset EVAL_DATASET        eval dataset path [needed for eval]
 --cover COVER                      path to csv tiles cover file, to filter tiles dataset on [optional]
 --classes_weights CLASSES_WEIGHTS  classes weights separated with comma or 'auto' [optional]
 --tiles_weights TILES_WEIGHTS      path to csv tiles cover file, with specials weights on [optional]
 --loader LOADER                    dataset loader name [if set override config file value]

Hyper Parameters [if set override config file value]:
 --bs BS                            batch size
 --lr LR                            learning rate
 --ts TS                            tile size
 --nn NN                            neurals network name
 --encoder ENCODER                  encoder name
 --optimizer OPTIMIZER              optimizer name
 --loss LOSS                        model loss

Training:
 --epochs EPOCHS                    number of epochs to train
 --resume                           resume model training, if set imply to provide a checkpoint
 --checkpoint CHECKPOINT            path to a model checkpoint. To fine tune or resume a training
 --workers WORKERS                  number of pre-processing images workers, per GPU [default: batch size]

Output:
 --saving SAVING                    number of epochs beetwen checkpoint saving [default: 1]
 --out OUT                          output directory path to save checkpoint and logs [required]
```
## neo vectorize
```
usage: neo vectorize [-h] --masks MASKS --type TYPE [--config CONFIG] --out
                     OUT

optional arguments:
 -h, --help       show this help message and exit

Inputs:
 --masks MASKS    input masks directory path [required]
 --type TYPE      type of features to extract (i.e class title) [required]
 --config CONFIG  path to config file [required, if no global config setting]

Outputs:
 --out OUT        path to output file to store features in [required]
```
