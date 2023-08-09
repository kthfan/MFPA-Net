# MFPA-Net
Implementation of MFPA-Net: An efficient deep learning network for automatic ground fissures extraction in UAV images of the coal mining area.

# Usage

#### Load model:
```python
from mfpanet import MFPANet

model = MFPANet(num_classes=19, pretrained=True)
pred_mask = model(img)
```

#### Classification task:
```bash
python main.py --img-dir <train image directory> --mask-dir <train mask directory> --save-path <model path>
```

# References
Jiang, X., Mao, S., Li, M., Liu, H., Zhang, H., Fang, S., ... & Zhang, C. (2022). MFPA-Net: An efficient deep learning network for automatic ground fissures extraction in UAV images of the coal mining area. International Journal of Applied Earth Observation and Geoinformation, 114, 103039. [url](https://www.sciencedirect.com/science/article/pii/S1569843222002278)
## Reference code:
Jungdae Kim. (2021). PyTorch implementation of Meta Pseudo Labels [url](https://github.com/kekmodel/MPL-pytorch/blob/main/models.py).

Fisher Yu, Vladlen Koltun, & Thomas Funkhouser (2017). Dilated Residual Networks. In Computer Vision and Pattern Recognition (CVPR) [url](https://github.com/fyu/drn/blob/master/drn.py).

https://github.com/VainF/DeepLabV3Plus-Pytorch

https://github.com/mlyg/unified-focal-loss
