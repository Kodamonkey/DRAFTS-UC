---
license: mit
tags:
- Astronomy
- Classification
- Object Detection
---

<div align="center">

<h1 style="font-size:40px;font-weight:bold">DRAFTS</h1>

_✨ Deep learning-based RAdio Fast Transient Search pipeline✨_

<img src="https://counter.seku.su/cmoe?name=APOD&theme=r34" /><br>
</div>

## <div align="center">Description</div>

Here is the model repository for the Deep learning-based RAdio Fast Transient Search pipeline ([DRAFTS](https://github.com/SukiYume/DRAFTS)).

We invite you to stay tuned for updates on the remaining components and different versions of the models.

## <div align="center">Usage</div>

There are four `.pth` files in this repository.

### Object Detection

The files starting with `cent` are the trained model checkpoints for the object detection models in the [DRAFTS](https://github.com/SukiYume/DRAFTS) project.

To load the object detection model

```python
import torch
from centernet_model import centernet

base_model = 'resnet18' # 'resnet50'
model = centernet(model_name=base_model)
model.load_state_dict(torch.load('cent_{}.pth'.format(base_model)))
model.eval()
```

### Binary Classification

The files starting with `class` are the trained model checkpoints for the classification models in the [DRAFTS](https://github.com/SukiYume/DRAFTS) project.

To load the classification detection model

```python
import torch
from binary_model import BinaryNet

base_model = 'resnet18' # 'resnet50'
model = BinaryNet(base_model, num_classes=2)
model.load_state_dict(torch.load('class_{}.pth'.format(base_model)))
model.eval()
```


## <div align="center">Contributing</div>

We welcome contributions to the DRAFTS project! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request on our [GitHub repository](https://github.com/SukiYume/DRAFTS).


## <div align="center">Contact</div>
For any questions or inquiries, please contact us at ykzhang@nao.cas.cn or ykzhang@escape.ac.cn

<div align="center">
✨ Thank you for using DRAFTS! ✨
</div>
