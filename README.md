# -PyTorch_YOLOv4-tiny
This project is based on WongKinYiu / PyTorch_YOLOv4 with some modification

- You can use this project to train 416x416 YOLOv4-tiny.

[GitHub Issues](https://github.com/WongKinYiu/ScaledYOLOv4/issues/41)

### 12/1更新：

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
透過[u3_preview](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u3_preview?rgh-link-date=2020-11-24T04%3A40%3A32Z)版本經由以下指令訓練300個Epochs並進行testing得到的AP結果：
```
CUDA_VISIBLE_DEVICES=0 python train.py --data coco2017.data --cfg yolov4-tiny.cfg --weights 'yolov4-tiny.conv.29' --name yolov4-tiny --img 416 416 416
python test.py --data coco2017.data --cfg yolov4-tiny.cfg --weights weights/best.pt --img 416 --batch-size 8
```
darknet版本測出是40.2
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.221
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
```