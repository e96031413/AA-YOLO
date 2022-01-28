# Attention ALL-CNN Twin Head YOLO (AA -YOLO)

Official implementation of **Improving Tiny YOLO with Fewer Model Parameters**, IEEE BigMM 2021

### Abstract
With the rapid development of convolutional neural networks (CNNs), there are a variety of techniques that can improve existing CNN models, including attention mechanisms, activation functions, and data augmentation. However, integrating these techniques can lead to a significant increase in the number of parameters and FLOPs. Here, we integrated Efficient Channel Attention Net(ECA-Net), Mish activation function, All Convolutional Net (ALL-CNN), and a twin detection head architecture into YOLOv4-tiny, yielding an AP 50 of 44.2% on the MS COCO 2017 dataset. The proposed Attention ALL-CNN Twin Head YOLO (A 2 -YOLO) outperforms the original YOLOv4-tiny on the same dataset by 3.3% and reduces the model parameters by 7.26%. Source code is at https://github.com/e96031413/AA-YOLO

### Note
This project is based on [WongKinYiu/PyTorch_YOLOv4 u3_preview branch](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u3_preview) with some modification

The AA-YOLO architecture file is at [AA-YOLO-twin-head.cfg](https://github.com/e96031413/PyTorch_YOLOv4-tiny/blob/main/cfg/AA-YOLO-twin-head.cfg)

- You can use this project to train 416x416 YOLOv4-tiny.
[GitHub Issues](https://github.com/WongKinYiu/ScaledYOLOv4/issues/41)

- View our experiment environment infos [here](https://github.com/e96031413/PyTorch_YOLOv4-tiny/blob/main/experiment-info.md)

### Development Log
<details><summary> <b>Expand Development Log</b> </summary>

### 2021/03/04更新：
使用test.py針對Cross-Stitch架構進行AP測試時，必須到test.py的第43行將model.fuse()功能關閉
```
#model.fuse()
```

### 2021/02/08更新：
新增detect.py計算FPS功能 (detect.py的第8行、138~140行、171行)
```
# line 8
FPS_avg = []

# line 138~140
from statistics import mean 
print('%sDone. (FPS:%.1f)' % (s, 1/(t2 - t1)))
FPS_avg.append(1/(t2-t1))

# line 171
print('Avg FPS: (%.1f)' % (mean(FPS_avg)))
```

解決test.py執行時遇到numpy必須要1.17版本及No module named 'pycocotools'的方法:
```bash
#移除所有numpy
pip uninstall numpy
#安裝1.17版本numpy
pip install numpy==1.17
#安裝pycocotools
pip install pycocotools
#執行test.py
python test.py --data coco2017.data --cfg yolov4-tiny.cfg --weights yolov4-tiny.pt --img 416 --augment
```

### 2021/01/10更新：
新增支援yolov4.conv.137的Pre-trained weight功能(在model.py第456~456行)
```
    elif file == 'yolov4.conv.137':
        cutoff = 137
```

### 2020/12/29更新：
新增YOLOv4-tiny的RouteGroup功能

[Feature-request: YOLOv4-tiny #1350](https://github.com/ultralytics/yolov3/issues/1350#issuecomment-651602149)

新增ReLU6(在utils/layers.py)、ReLU(在utils/layers.py)、DepthWise Convolution(在models.py)、ShuffleNetUnit(在models.py)

**如何透過[u5版本](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u5)的yaml檔案進行backbone修改？**

[Yolov4 with Efficientnet b0-b7 Backbone](https://shihyung1221.medium.com/yolov4-with-efficientnet-b0-b7-backbone-529d0ce67cf0)

[Yolov4 with MobileNet V2/V3 Backbone](https://shihyung1221.medium.com/yolov4-with-mobilenet-v2-v3-backbone-c18c0f10bc29)

[Yolov4 with Resnext50/ SE-Resnet50 Backbones](https://shihyung1221.medium.com/yolov4-with-resnext50-se-resnet50-backbones-c324242c48f4)

[Yolov4 with Resnet Backbone](https://shihyung1221.medium.com/yolov4-with-resnet-backbone-eb141b6e79ca)

[Yolov4 with VGG Backbone](https://shihyung1221.medium.com/yolov4-with-vgg-backbone-ae0cedab4f0f)

### 2020/12/28更新：
新增以下四篇Paper的程式碼(在models.py、utils/layers.py)

SE Block paper : [arxiv.org/abs/1709.01507](arxiv.org/abs/1709.01507)

CBAM Block paper : [arxiv.org/abs/1807.06521](arxiv.org/abs/1807.06521)

ECA Block paper : [arxiv.org/abs/1910.03151](arxiv.org/abs/1910.03151)

Funnel Activation for Visual Recognition : [arxiv.org/abs/2007.11824](arxiv.org/abs/2007.11824)


### 2020/12/1更新：
修改了PyTorch_YOLOv4的u3_preview當中，models.py第355行(支援pre-trained weight)、train.py第67行(支援32倍數的解析度)、dataset.py第262及267行(處理dataset相對路徑的問題)
models.py line 355
```python
def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15
    elif file == 'yolov4-tiny.conv.29':
        cutoff = 29
	.
	.
	.
	.
```
train.py line 67
```python
    gs = 32  # (pixels) grid size # 原本gs = 64改成32
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
```
utils/dataset.py  line 262 and 267
```python
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False):
        path = str(Path(path))  # os-agnostic
        parent = str(Path(path).parent) + os.sep                              # add this
        assert os.path.isfile(path), 'File not found %s. See %s' % (path, help_url)
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in img_formats]
            self.img_files = [x.replace('./', parent) if x.startswith('./') else x for x in self.img_files]    # add this
```
</details>

## Experiment Environment

Our environment setting on [https://www.twcc.ai/](TWCC)
```
NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0
python 3.6.9 
PyTorch 1.6.0
Torchvision 0.7.0
tensorflow-20.06-tf2-py3:tag1614508696283
numpy 1.17.0
```

## Important note!!!
2021/03/31 update:

**Be careful to use NAS (Network Attached Storage) to store the coco dataset, it might slow down the tranining speed. Store your dataset locally.**

## Requirements

```
# Install mish-cuda if you want to use fewer GPU memory during training (from 12G -> 8G) 
pip install git+https://github.com/thomasbrandon/mish-cuda/
pip install -r requirements.txt
```

## Download COCO 2017 dataset
```
sh data/get_coco.sh
```

## Training

```
CUDA_VISIBLE_DEVICES=0 python train.py --data coco2017.data --cfg AA-YOLO-twin-head.cfg --weights 'yolov4-tiny.conv.29' --name yolov4-tiny --img 416
```

## Testing

```
CUDA_VISIBLE_DEVICES=0 python test.py --data coco2017.data --cfg AA-YOLO-twin-head.cfg --weights yolov4-tiny.pt --img 416 --augment
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 python detect.py --weights AA-YOLO-twin-head.pt --img 416 --source file.mp4  # video
                                                                                    file.jpg  # image 
```

## Training Result Visualization
```
python -c from utils import utils;utils.plot_results().
```

## Citation
```
@inproceedings{9643269,  
    title={Improving Tiny YOLO with Fewer Model Parameters},
    author={Liu, Yanwei and Ma, Ching-Wen},  
    booktitle={2021 IEEE Seventh International Conference on Multimedia Big Data (BigMM)},   
    pages={61-64},
    year={2021},
    doi={10.1109/BigMM52142.2021.00017}}
```

## Acknowledgements

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [HaloTrouvaille/YOLO-Multi-Backbones-Attention](https://github.com/HaloTrouvaille/YOLO-Multi-Backbones-Attention/tree/1f425d379783b6d132b44f14ecfd251d8e2448fa)
* [SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone](https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone)
