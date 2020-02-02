<h1 align='center'>Neat-EO.pink</h1>
<h2 align='center'>Computer Vision framework for GeoSpatial imagery, at scale</h2>

<p align=center>
  <img src="https://pbs.twimg.com/media/DpjonykWwAANpPr.jpg" alt="Neat-EO.pink buildings segmentation from Imagery" />
</p>



Purposes:
---------
- DataSet Quality Analysis
- Change Detection highlighter
- Features extraction and completion


Main Features:
--------------
- Provides several command line tools, you can combine together to build your own workflow
- Follows geospatial standards to ease interoperability and data preparation 
- Build-in cutting edge Computer Vision model, Data Augmentation and Loss implementations (and allows to replace by your owns)
- Support either RGB and multibands imagery, and allows Data Fusion 
- Web-UI tools to easily display, hilight or select results (and allow to use your own templates)
- High performances
- Eeasily extensible by design




<img alt="Draw me Neat-EO.pink" src="https://raw.githubusercontent.com/datapink/neat-eo.pink/master/docs/img/readme/draw_me_neat_eo.png" />


 
Documentation:
--------------
### Tutorial:
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/101.md">Learn to use Neat-EO.pink, in a couple of hours</a>

### Config file:
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/config.md">Neat-EO.pink configuration file</a>

### Tools:

- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-cover">`neo cover`</a> Generate a tiles covering, in csv format: X,Y,Z
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-download">`neo download`</a> Downloads tiles from a remote server (XYZ, WMS, or TMS)
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-extract">`neo extract`</a> Extracts GeoJSON features from OpenStreetMap .pbf
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-rasterize">`neo rasterize`</a> Rasterize vector features (GeoJSON or PostGIS), to raster tiles
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-subset">`neo subset`</a> Filter images in a slippy map dir using a csv tiles cover
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-tile">`neo tile`</a> Tile raster coverage
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-dataset">`neo dataset`</a> Perform checks and analyses on Training DataSet
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-train">`neo train`</a> Trains a model on a dataset
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-export">`neo export`</a> Export a model to ONNX or Torch JIT
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-predict">`neo predict`</a> Predict masks, from given inputs and an already trained model
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-compare">`neo compare`</a> Compute composite images and/or metrics to compare several XYZ dirs
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-vectorize">`neo vectorize`</a> Extract simplified GeoJSON features from segmentation masks
- <a href="https://github.com/datapink/neat-eo.pink/blob/master/docs/tools.md#neo-info">`neo info`</a> Print Neat-EO.pink version informations

### Presentations slides:
  - <a href="http://www.datapink.com/presentations/2020-fosdem.pdf">@FOSDEM 2020</a>





Installs:
--------

### With PIP:
```
pip3 install Neat-EO.pink
```

### With Ubuntu 19.10, from scratch:

```
# Neat-EO.pink [mandatory]
sudo sh -c "apt update && apt install -y build-essential python3-pip"
pip3 install Neat-EO.pink && export PATH=$PATH:~/.local/bin

# NVIDIA GPU Drivers [mandatory]
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/435.21/NVIDIA-Linux-x86_64-435.21.run
sudo sh NVIDIA-Linux-x86_64-435.21.run -a -q --ui=none

# Extra CLI tools [used in tutorials]
sudo apt install -y gdal-bin osmium-tool

# HTTP Server [for WebUI rendering]
sudo apt install -y apache2 && sudo ln -s ~ /var/www/html/neo
```


### NOTAS: 
- Requires: Python 3.6 or 3.7
- GPU with VRAM >= 8Go is mandatory
- To test Neat-EO.pink install, launch in a new terminal: `neo info`
- If needed, to remove pre-existing Nouveau driver: ```sudo sh -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf && update-initramfs -u && reboot"```




Architecture:
------------

Neat-EO.pink use cherry-picked Open Source libs among Deep Learning, Computer Vision and GIS stacks.

<img alt="Stacks" src="https://raw.githubusercontent.com/datapink/Neat-EO.pink/master/docs/img/readme/stacks.png" />



GeoSpatial OpenDataSets:
------------------------
- <a href="https://github.com/chrieke/awesome-satellite-imagery-datasets">Christoph Rieke's Awesome Satellite Imagery Datasets</a>
- <a href="https://zhangbin0917.github.io/2018/06/12/%E9%81%A5%E6%84%9F%E6%95%B0%E6%8D%AE%E9%9B%86/">Zhang Bin, Earth Observation OpenDataset blog</a> 

Bibliography:
-------------

- <a href="https://arxiv.org/abs/1912.01703">PyTorch: An Imperative Style, High-Performance Deep Learning Library</a>
- <a href="https://arxiv.org/abs/1505.04597">U-Net: Convolutional Networks for Biomedical Image Segmentation</a>
- <a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a>
- <a href="https://arxiv.org/abs/1806.00844">TernausNetV2: Fully Convolutional Network for Instance Segmentation</a>
- <a href="https://arxiv.org/abs/1705.08790">The Lovász-Softmax loss: A tractable surrogate for the optimization of the IoU measure in neural networks</a>
- <a href="https://arxiv.org/abs/1809.06839">Albumentations: fast and flexible image augmentations</a>










Contributions and Services:
---------------------------

- Pull Requests are welcome ! Feel free to send code...
  Don't hesitate either to initiate a prior discussion via <a href="https://gitter.im/DataPink/Neat-EO">gitter</a> or ticket on any implementation question.
  And give also a look at <a href="https://github.com/datapink/robosat.pink/blob/master/docs/makefile.md">Makefile rules</a>.

- If you want to collaborate through code production and maintenance on a long term basis, please get in touch, co-edition with an ad hoc governance can be considered.

- If you want a new feature, but don't want to implement it, <a href="http://datapink.com">DataPink</a> provide core-dev services.

- Expertise, assistance and training on Neat-EO.pink are also provided by <a href="http://datapink.com">DataPink</a>.

- And if you want to support the whole project, because it means for your own business, funding is also welcome.


### Requests for funding:

We've already identified several new features and research papers able to improve again Neat-EO.pink,
your funding would make a difference to implement them on a coming release:

- Increase (again) prediction accuracy :
  - on low resolution imagery
  - even with few labels (aka Weakly Supervised)
  - feature extraction when they are (really) close (aka Instance Segmentation)

- Add support for :
  - Linear features extraction
  - Time Series Imagery
  - StreetView Imagery
  
- Improve (again) performances




Authors:
--------
- Olivier Courtin <https://github.com/ocourtin>
- Daniel J. Hofmann <https://github.com/daniel-j-h>



Citing:
-------
```
  @Manual{,
    title = {Neat-EO.pink} Computer Vision framework for GeoSpatial Imagery},
    author = {Olivier Courtin, Daniel J. Hofmann},
    organization = {DataPink},
    year = {2020},
    url = {http://Neat-EO.pink},
  }
```
