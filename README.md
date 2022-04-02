# README

# 업스테이지 수학 수식 OCR 모델

*수식은 여러 자연과학 분야에서 어려운 개념들은 간단하고 간결하게 표현하는 방법으로서 널리 사용되어 왔습니다. Latex 또한 여러 과학 분야에서 사용되는 논문 및 기술 문서 작성 포맷으로서 현재까지도 널리 사용되고 있습니다.*

*수식 인식의 경우는, 기존의 광학 문자 인식 (optical character recognition)과는 달리 multi line recognition을 필요로 합니다. 우리가 알고 있는 분수 표현, 극한, 시그마 와 같은 표현만 보더라도 수식 인식 문제에서는 multi line recognition이 중요한 문제라는 것을 알 수 있습니다*

*수식인식 문제는 단순하게 수식을 인식하는 문제로도 볼 수 있지만 기존 single line recognition 기반의 OCR이 아닌 multi line recognition을 이용하는 OCR task로도 인식할 수 있습니다. Multi line recognition 이란 관점에서 기존 OCR과는 차별화되는 task라고 할 수 있습니다.*

*Mathematics has been widely used as a simple and concise way of expressing difficult concepts in various fields of natural science. Latex is also widely used to date as a paper and technical writing format used in various scientific fields.*

*Equation recognition requires multi-line recognition, unlike conventional optical character recognition. Just by looking at expressions such as fractional expressions, limits, and sigma that we know, we can see that multi-line recognition is an important issue in the equation recognition problem.*

*The equation recognition problem can be seen as a simple equation recognition problem, but it can also be recognized as an OCR task using multi-line recognition rather than an existing single-line recognition-based OCR. From the perspective of multi-line recognition, it is a task that is different from the existing OCR.*

## Requirements

- Python 3
- [PyTorch](https://pytorch.org/)

All dependencies can be installed with PIP.

```
pip install tensorboardX tqdm pyyaml psutil
```

현재 검증된 GPU 개발환경으로는 - `Pytorch 1.0.0 (CUDA 10.1)` - `Pytorch 1.4.0 (CUDA 10.0)` - `Pytorch 1.7.1 (CUDA 11.0)`

## Supported Data

- [Aida](https://www.kaggle.com/aidapearson/ocr-data) (synthetic handwritten)
- [CROHME](https://www.isical.ac.in/~crohme/) (online handwritten)
- [IM2LATEX](http://lstm.seas.harvard.edu/latex/) (pdf, synthetic handwritten)
- [Upstage](https://www.upstage.ai/) (print, handwritten)

## Models

[SATRN](https://github.com/clovaai/SATRN)을 사용 했습니다. 공식적으로 Pytorch version을 지원하지 않기에 직접 구현하기로 하였습니다.
Baseline 코드로 어느 정도 뼈대를 제공 했지만 논문에서 가장 강조한 부분들이 빠져 있었습니다.

We use [SATRN](https://github.com/clovaai/SATRN). It did not support Pytorch version so I have to make our own code.

Adaptive 2D positional encoding 과Locality-aware feedforward layer 부분인데요.
이부분을 논문과 같이 구현해 어느 정도 성능이 오르는지 확인해 보았습니다.

|Positional Encoding |Feedforward layer| Public LB |
|:---|:---| :---|
|2D-Concat |FC| 0.7698|
| A2DPE|Conv | 0.7717


## Image Sizes

128x128, 64x256

![README%20eeb1c0530360423a914964ca597bd7c5/ratio.png](README%20eeb1c0530360423a914964ca597bd7c5/ratio.png)

이미지의 평균 비율이 1:4인점을 볼 수 있는데요. 
이 결과를 바탕으로 이미지를 64x256으로 resize 해주었습니다. 아울러 이미지 크기를 크게 해줄수록 성능이
올라가는 것을 확인할 수 있었지만, 저희 환경에서 1:4 비율을 유지하면서 크기를 늘릴 수 없었습니다.
0.06 정도의 성능이 올랐습니다.

## Augmentations

I used the following [Albumentations](https://github.com/albu/albumentations):

```
A.Compose(
        [
            A.Resize(options.input_size.height, options.input_size.width)
            A.Normalize(
                         mean=(0.6156),
                         std=(0.1669), max_pixel_value=255.0, p=1.0
                            ),
            ToTensorV2()
        ]

    )
```

복잡한 augmentation 보다 위에 볼 수 있는 간단한 augmentation 성능이 더 좋았습니다.

## TTA

Test datasets에서도 세로가 가로보다 긴 이미지들이 있을거라 예상하여, 그러한 이미지들을 90° 그리고 -90°로 rotate 하여 결과를 내보았습니다. 90°회전을 시켰을 때가 가장 성능이 좋았습니다.
0.003 소폭 상승 했습니다.

## Final Score

|Public LB |Private LB|
|:---|:---|
|0.7750 |0.55| 

## Discussion

### Improvements

Next time I would like to:

    Other Schedulers
    Other Loss function
    Swin Transformer



## 모든 데이터는 팀 저장소에서 train-ready 포맷으로 다운 가능하다.

```
[dataset]/
├── gt.txt
├── tokens.txt
└── images/
    ├── *.jpg
    ├── ...
    └── *.jpg
```

## Usage

### Training

```
python train.py
```

### Evaluation

```
python evaluate.py
```
