## BaMix:BALANCED MIXUP LOSS FOR LONG-TAILED VISUAL RECOGNITION
### Dataset

- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.
- ImageNet-LT. We followed [CAM](https://github.com/zhangyongshun/BagofTricks-LT) to generate long-tailed ImageNet.

### Main requirements
```bash
Python==3.7
pytorch==1.7.1
torchaudio==0.10.0
torchvision==0.11.1
tensorboardX==2.4
matplotlib==3.4.3
scikit-learn==1.0.1
opencv-python==4.5.3.56
seaborn==0.11.2
numpy==1.21.4
pandas==1.1.5
pillow==8.4.0
six==1.16.0
tqdm==4.62.3
```

### file structure
    BaMix
      ├── data
      │     ├── ImageNet_LT
      │     │     ├── ImageNet_LT_train.json
      │     │     ├── ImageNet_LT_val.json
      ├── lib
      ├── main
    dataset
      ├── imbalance_cifar
      │     ├── cifar-10-batches-py
      │     ├── cifar-100-python
      ├── ImageNet    
      │     ├── train
      │     │     ├── n01440764
      │     │     │       ├── n01440764_18.JPEG
      │     │     │       ├── n01440764_36.JPEG
      │     │     │       ├── ......
      │     │     ├── n01443537
      │     │     │       ├── ......
      │     │     ├── ......
      │     ├── val
      │     │     ├── ILSVRC2012_val_00000001.JPEG
      │     │     ├── ILSVRC2012_val_00000002.JPEG
      │     │     ├── ......


### Training 

We provide several training examples with this repo:

- To train cifar10-LT with imbalance ratio of 100

```bash
python cifar_train.py --dataset cifar10 -im 0.01 --device 0
```

- To train cifar100-LT with imbalance ratio of 200

```bash
python cifar_train.py --dataset cifar100 -im 0.005 --device 0
```

- To train ImageNet-LT with architecture of resnet10 and batch size of 256.

```bash
python ImgNet_train.py -a resnet10i -b 256  --devices 0  --mixepoch 160
```

- To train ImageNet-LT with architecture of resnet50 and batch size of 128 on two gpus.

```bash
python ImgNet_train.py -a resnet50i -b 128  --devices 0,1 --mixepoch 160
```

-To train iNaturalist 2018

```bash
python iNaturalist_train.py -a resnet50i -b 128 --devices 0
```
### Testing

#### Make sure that the name of the folder conforms to our naming convention, which can be automatically generated through training.

- To test cifar-10/100-LT

```bash
python test.py --resume ../checkpoint/cifar10_resnet32_STM_DRW_0_0.01_alpha_1.0_maxm_0.5_mixepoch_360/best.pth.tar
```
