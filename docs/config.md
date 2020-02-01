## config.toml
```
# Neat-EO.pink Configuration


# Input channels configuration
# You can, add several channels blocks to compose your input Tensor. Order is meaningful.
#
# name:		dataset subdirectory name
# bands:	bands to keep from sub source. Order is meaningful

[[channels]]
  name   = "images"
  bands = [1, 2, 3]


# Output Classes configuration
# Nota: available colors are either CSS3 colors names or #RRGGBB hexadecimal representation.
# Nota: special color name "transparent" could be use on a single class to apply transparency
# Nota: default weight is 1.0 for each class, or 0.0 if a transparent color one.

[[classes]]
  title = "Background"
  color = "transparent"

[[classes]]
  title = "Building"
  color = "deeppink"

[[classes]]
  title = "Road"
  color = "deepskyblue"


[model]
  # Neurals Network name
  nn = "Albunet"

  # Encoder name
  encoder = "resnet50"

  # Dataset loader name
  loader = "SemSeg"

  # Model internal input tile size [W, H]
  #ts = [512, 512]


[train]
  # Pretrained Encoder
  #pretrained = true

  # Batch size
  #bs = 4

  # Data Augmentation to apply, to whole input tensor, with associated probability
  da = {name="RGB", p=1.0}

  # Loss function name
  loss = "Lovasz"

  # Eval Metrics
  metrics = ["IoU", "MCC", "QoD"]

  # Optimizer, cf https://pytorch.org/docs/stable/optim.html
  #optimizer = {name="Adam", lr=0.0001}
```
