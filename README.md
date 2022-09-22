# README

# 업스테이지 수학 수식 OCR 모델

![image](https://user-images.githubusercontent.com/51700219/147439435-30a348cc-14b7-45d1-85c3-0dcc23c1a666.png)

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

We use [SATRN](https://github.com/clovaai/SATRN). But it did not support Pytorch version so I have to make our own code.

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



# 수식인식기 프로젝트 소개
![image](https://user-images.githubusercontent.com/51700219/147439435-30a348cc-14b7-45d1-85c3-0dcc23c1a666.png)
 - 위의 예시에서 처럼 어떤 수식이 적힌 이미지를 input으로 받으면 이 수식에 대한 latex문법 표현식을 출력해주는 모델을 만드는 프로젝트이다.
 - 위의 예시에서의 수식이미지는 컴퓨터로 인쇄된 수식이미지 이지만 실제로는 손으로 쓴 수식이미지도 사용되었다.
 - 이 프로젝트에서 사용한 데이터에서는 이미 text localization이 수행된 데이터들이다. 즉, 이 프로젝트에서는 text recognition만 한다.

# SATRN 모델 소개
![image](https://user-images.githubusercontent.com/51700219/147440473-51748720-1f0b-475c-8601-04f9fa7f406a.png)
 - input 이미지는 CNN을 거쳐서 feature map이 된 후 Transformer 구조를 가진 네트워크를 통과하여 최종 output이 된다.
 - Transformer구조를 가진 부분은 기존 Transformer와 구조가 유사하지만 input으로 자연어가 아닌 feature map을 받으므로 이 부분을 고려하여 기존 Transformer와는 조금씩 다른 부분들을 가지고 있다.
 - SATRN 논문리뷰 : https://github.com/bcaitech1/p4-fr-9-googoo/discussions/4

# 평가 metric 
![image](https://user-images.githubusercontent.com/51700219/147821386-81181bbb-8969-4749-bd39-e0f00115cd76.png)
 - metric으로는 sentence accuracy를 사용한다.
 - 예측 문장과 정답 문장이 완벽히 일치하는 데이터의 비율이다.

# 실험기록
#### 대회서버가 불안정하여 리더보드에 제출이 안되는 이유로 모든 실험은 리더보드 점수가 아닌 validation data에 대한 점수를 기반으로 진행 하였다. 다만 다른 분들의 말에 따르면 validation 점수와 리더보드 점수가 거의 똑같다고 한다. 


### 5/27 ~ 5/29
 - 주어진 baseline을 그대로 사용하여 30epoch단위로 끊어서 실행하였다. 아래는 validation data에 대한 sentence accuracy이다.

![image](https://user-images.githubusercontent.com/51700219/120093544-f911b400-c155-11eb-99ca-ee4af7e4ec0f.png)
 - 30epoch 이후 부터 학습을 이어서 진행해 보았다. 아래는 이에 대한 sentnece accuracy이다.
![image](https://user-images.githubusercontent.com/51700219/120093580-4857e480-c156-11eb-9b57-b88b74ed6fb0.png)
 - overfit이 계속 일어나서 중간에 중단하였다. 30epoch정도가 적절해 보인다.
 - 다만 아래 그림(첫 30epoch동안의 learning rate)에서처럼 총 epoch수를 기준으로 learning rate가 cyclic하게 조절되는 스케쥴러를 사용하였기 때문에 처음부터 총 epoch수를 30이 아니라 50이나 다른 수로 잡는다면 결과가 달라질 수 있다. 하지만 30epoch씩 두번 하는건 과적합을 일으킨다. 두번째 30에폭에서는 learning rate가 다시 증가하게 되고 learning rate가 너무 크면 loss가 수렴하지 않고 튈 수 있기 때문이다.

![image](https://user-images.githubusercontent.com/51700219/120093655-ad133f00-c156-11eb-98b5-4434f6add32d.png)
 - 나중에 더 해보니 처음부터 50에폭으로 잡으면 accuracy가 더 올라간다. 60에폭으로 하면 더더 올라간다.
 - epoch을 크게할 수록 learning rate가 더 천천히 증가하다가 천천히 감소하기 때문에 더 안정적으로 학습이 가능한 것으로 보인다.

### 5/29 ~ 5/30
 - 베이스라인으로 받은 STARN 코드가 잘못되어 있는 것 같다. Transformer구조의 decoder부분의 구현이 잘못되어 있다. 
 - 관련 issue 링크: https://github.com/bcaitech1/p4-fr-9-googoo/issues/5

### 5/31 ~ 6/2
 - 코드 구현 관련된 부분을 수정후 batch size = 34, epoch = 50으로 하여 학습.

![image](https://user-images.githubusercontent.com/51700219/120580288-e2c06c80-c463-11eb-9e2a-a8bdd5dbcb33.png)
 - 수정 후 validation sentence accuracy = 0.6973으로 수정 전(0.6887)과 비교하여 약 1%정도 올랐다.

### 6/5 ~ 6/7 
 - 코드에서 loss함수를 계산할 때 padding토큰에 대한 loss를 무시하는 부분을 추가하였다.
 - 기존 베이스라인 코드에서는 CrossEntoropyLoss에 패딩토큰에 대한 loss까지도 계산하는 형태였는데 이 부분이 이해가 안되서 왜 이렇게 하는지 질문을 해보니 그냥 오타였다.
 - padding 토큰에 대한 loss는 무시함으로써 End Of Sentence 토큰까지의 토큰들을 잘 예측하는 것에 더 집중할 수 있기 때문에 성능이 더 오를것이라 생각하였다.
 - 다른 팀원분께서 실험을 해주셨는데 아무것도 안한 기본 베이스라인 코드에서 padding토큰에 대한 loss를 무시하는 코드만 추가하였더니 sentence accuracy가 0.6825에서 0.7037로 올랐다.

### 6/2 ~ 6/10
 - 베이스라인 코드에는 encoder의 feedforward network로 fully connected가 사용되었다(아래 그래프의 SATRN_v4).
 - 원래 SATRN논문에서 사용한 방식대로 depth-wise separable convolution으로 교체하여 학습해보았다(아래 그래프의 SATRN_v5).

![image](https://user-images.githubusercontent.com/51700219/120734943-b02a7880-c524-11eb-94d9-e2fd98c8bd79.png)
 - 위의 그래프에서 처럼 처음에는 accuracy가 더 높게 나오다가 끝에 가서는 기존 모델(v4)보다 더 낮게 되었다(v4의 accuracy : 0.6973, v5의 accuracy:0.689).

![image](https://user-images.githubusercontent.com/51700219/120735139-05ff2080-c525-11eb-91ee-e18af0d02b7e.png)
 - 위의 그래프에서 처럼 train data에 대한 sentence accuracy는 v5가 v4보다 항상 높았다.
 - 과적합이라 판단하고 drop out rate를 기존의 0.1에서 0.3으로 높여서 다시 실험해보았다.

![image](https://user-images.githubusercontent.com/51700219/121015322-d32f8380-c7d5-11eb-88cf-ea7a914f174d.png)
 - 위의 그래프에서 처럼 drop out을 0.3으로 준 모델(v6)의 validation sentence accuracy가 v5(drop out 0.1)보다 안좋게 나왔다.
 - drop out을 너무 과하게 준 것으로 보이므로 0.2로 줄여서 다시 실험해 보았다.

![image](https://user-images.githubusercontent.com/51700219/121016056-a62fa080-c7d6-11eb-840e-38f96da23ee9.png)
 - 위의 그래프에서 처럼 drop out을 0.2로 준 모델(v7)이 v5보다 0.003정도 더 높게 나왔다. 그러나 여전히 v4보다는 작다.
 - drop out을 좀 더 조절해주면 좀 더 accuracy를 높일 수도 있을 것 같았지만 시간이 너무 오래 걸려서 하지 않았다. 한다고 하더라도 v4보다 크게 높은 점수를 얻을 수 있을 것 같진 않았다.
 - 이에 대해 가설을 하나 세웠는데 현재 가지고 있는 train 데이터로는 validation 데이터에게 까지 적용할 수 있는 global한 feature를 더 이상 학습할 수 없는게 아닐까 하는 것이다. 즉, 학습할 수 있는 global feature는 전부 학습해서 더 이상 남아있지 않아서 더 좋은 모델을 써도 점수가 안오르거나 심지어 오버피팅으로 점수가 낮아지는게 아닐까?
 - 그래서 depth-wise separable convolution을 적용해도 학습할 수 있는 global feature가 더 이상 없으니 depth-wise separable convolution을 적용하나 안하나 결과가 비슷한 것 같다. 다만 train data에만 적용되는 local feature는 train loss가 0이 될 때까지 계속 학습할 수 있으므로 depth-wise separable convolution을 적용했을 때의 train sentence accuracy가 더 높은 것이다.
 - 이를 확인해보려면 데이터가 더 있어야 할 것 같다. fully connected로는 global feature를 더 학습하는 것에 한계가 있지만 depth-wise separable convolution으로는 fully connected보다 더 많은 global feature를 학습할 수 있을만큼 데이터가 많다면 확인해 볼 수 있을 것이다.
 - 마침 다른 팀원분께서 128x128로 resize해서 사용하던 이미지 데이터를 64 x 256으로 resize 해보는 실험을 해보신 결과 validation data에 대한 점수를 크게 올려 주셨다(validation sentence accuracy가 0.7401까지 올라갔다). validation data에 대한 점수가 올랐다는 것은 이렇게 resize 된 데이터에는 학습할 수 있는 global feature가 기존보다 더 많다는 것이므로 데이터를 늘리지 않고도 resize만 해서 위의 가설을 실험해 볼 수 있을 것 같다.

![image](https://user-images.githubusercontent.com/51700219/121282204-0ed06780-c914-11eb-8d26-44f55edf468f.png)
 - 64 x 256으로 resize하고 depth-wise separable convolution적용하여 실험 해보니 validation sentence accuracy가 기존의 0.7401에서 0.7527까지 올라갔다 

### 6/10 ~ 6/11
 - CBAM 논문을 공부해보니 여기서는 attention을 계산 할 때 spatial attention만 한게 아니라 아래 그림처럼 channel attention도 하여 점수를 올렸다고 한다.

![image](https://user-images.githubusercontent.com/51700219/122502865-66688480-d032-11eb-9ee3-c0db8798f007.png)

 - CBAM에서는 channel attention과 spatial attention을 번갈아 가면서 적용하는 방식으로 score를 올렸다고 한다.
 - SATRN도 image를 input으로 받고 attention을 사용하는 모델이므로 CBAM의 channel attention을 SATRN 모델에도 적용하면 성능 개선에 도움이 될 수 있을것이라 생각하였다.
 - attention을 계산하는 방식은 꽤 다양한데 CBAM 에서는 Transformer에서와는 다른 방식의 attenton을 사용하고 있다.
 - CBAM에서의 attention 계산방식이 파라미터 수가 더 적다. 현재 사용중인 모델을 기준으로 Transformer의 self attention으로 channel attention을 적용할 경우 필요한 파라미터수는 한번 channel attention을 할 때마다 대략 36만개(query, key, value에 대한 linear layer가 따로 존재하고 output linear layer도 존재)인 반면 CBAM에서의 방식을 쓰면 한번 channel attention을 할 때마다 20000개 정도의 파라미터가 필요하다(CBAM에서는 하나의 channle attn 모듈이 가지는 파라미터는 shared mlp밖에 없기 때문. mlp의 크기도 transformer의 self attention에서 linear layer의 크기보다 더 작다.). 다만 CBAM에서는 self attention을 하는 방식은 아니기 때문에 성능의 향상은 덜 할 것이라 생각된다.
 - 파라미터 수가 훨씬 작은 CBAM의 channel attention을 적용해 봐야 겠다.
 - encoder layer에서 self attention에 들어가기 바로 전에 channel attention을 적용하도록 구현하였다(하나의 encoder layer 안에서 channel attn -> self attn -> feedforward 순으로 적용).
 - 아래 그림에서 처럼 구현 하였다. 그림에서 w는 feature map의 값이고 p는 positional encoding 값이다.

![image](https://user-images.githubusercontent.com/51700219/122770670-a5bbfd00-d2e0-11eb-8c92-de406a091006.png)
 - 시간을 아끼기 위해 10에폭으로 실험을 하였다. 실험 결과 점수가 하락하였다(SATRN이 기존 모델, SATRN_channel_attn이 channel attention을 적용한 모델).

![image](https://user-images.githubusercontent.com/51700219/122852638-e866ef00-d34b-11eb-97d5-ef0c5c03f3d3.png)
 - 아무래도 positional encoding때문인 것 같다. CBAM에서의 channel attention은 feature map에 max,avg pooling을 바로 적용하는데 SATRN에서는 feature map에 2D positional encoding을 먼저 더한 후 여기서 pooling을 하기 때문에 더해진 positional encoding값이 pooling을 할 때 노이즈처럼 작용할 수 있을 것이다. 

#### positional encoding이 노이즈가 될 수도 있는 이유?
 - feature map에 positional encoding을 더한 상태로 pooling을 하면 position정보까지 고려해서 pooling이 되지 않을까?
 - 자세히 생각해보면 그렇지 않다.

![image](https://user-images.githubusercontent.com/51700219/123366633-7e508300-d5b3-11eb-8e1b-1bd10d701aea.png)
 - 위의 그림은 feature map에 positional encoding을 더하는 것을 나타내는 그림이다. shape이 (C,H,W)인 feature map에 같은 크기의 positional encoding을 더하고 있다.
 - positional_encoding[:,0,0]은 h=0, w=0인 위치의 position을 나타내는 벡터이다(그림의 파란색으로 빗금친 부분). 이런식으로 하나의 position을 하나의 vector로 나타낸다.
 - 여기에 channel attention을 적용할 경우 그림에서 feature map과 positional encoding을 더한 결과인 result의 각 채널에 대해서 pooling을 하게 된다. 즉, 그림에서 result에서 빨간색으로 빗금친 부분에 global pooling을 적용하여 하나의 값으로 만드는 식으로 총 C개의 값을 만들고 fully connected를 통과시켜 총 C개의 attention weight를 만들게 된다.
 - 이때 pooling을 하는 경우를 잘 생각해 봐야 한다. 예를 들어 그림의 빨간색 빗금친 부분에서 pooling을 한다면 이렇게 하는게 position정보를 고려해서 pooling을 하는 것이라 할 수 있을까?
 - 빨간색 빗금친 부분에는 각 position에 대한 positional encoding vector가 더해진 것이 아니다. 각 position에 대한 positional encoding vector의 첫번째 값들만 더해진 것이다.
 - 즉, 빨간색 빗금친 부분에 대해서 pooling을 한 결과는 각 position에 대한 정보를 고려해서 pooling을 한 것이 아니라 각 position에 대한 positional encoding vector의 첫번째 값만 고려해서 pooling을 한 것이다.
 - 첫번째 값만으로는 그 값이 어떤 position을 나타내는지를 알기는 힘들 것이다. 즉 더해진 첫번째 값들은 어떤 정보를 나타낸다고 보기 힘들다.
 - 그래서 노이즈라고 생각하였다.
 - 만에 하나 더해진 첫번째 값들이 position정보를 가지고 있다 해도 이상태에서 pooling을 하는 것 또한 맞지 않을 수 있다.
 - 예를 들어 max pooling을 한다면 max pooling은 주어진 kernel size내의 feature 값들 중 가장 큰값을 대표로 쓰는 방식인데 feature에 position정보가 더해진 상태에서 max pooling을 할 때 feature자체의 값은 더 작은데 position값이 더 커서 제일 큰 값이 될 수도 있기 때문에 문제가 될 수 있다. 이 경우 또한 position정보가 노이즈 처럼 작용하고 있는 것이다.
 - 다만 positional encoding값 자체는 -1~1사이의 작은 값이므로 크게 영향이 없을 수도 있다.
### 6/11 ~ 
 - channel attention을 적용하기 위해서는 positional encoding을 따로 적용할 수 있어야 한다.
 - 즉 positional encoding을 더하기 전의 feature map에  channel attention을 적용하고 그 뒤에 positional encoding을 적용해야 한다.
 - 그냥 생각해보면 positional encoding을 더하기 전에 channel attn을 하고 그 뒤에 positional encoding을 더하면 될 것 같지만(아래 그림에서 처럼), 이는 encoder layer가 1개 일때만 가능할 것 같다. encoder layer가 2개 이상이라면 이러한 방식을 적용해도 첫번째 layer의 output에는 positional encoding이 더해진 상태이기 때문에 두번째 layer에서의 channel attention은 positional encoding이 더해진 상태에서 적용되게 된다. 사실 이렇게 해도 될지 안될지는 좀 애매하기 때문에 실험을 해봐야 할 것 같다(결국 시간이 없어서 못했음).

![image](https://user-images.githubusercontent.com/51700219/122771523-7a85dd80-d2e1-11eb-84bc-d648570a1ade.png)

 - 관련된 논문으로 positional encoding을 따로 적용할 수 있는 방식을 소개하는 논문을 찾을 수 있었다.([https://arxiv.org/abs/2006.15595](https://arxiv.org/abs/2006.15595))
 - 논문 자체의 초점은 positional encoding을 따로 적용하는 것 보다는 현재의 positional encoding 적용방식 자체의 문제점을 지적하고 이를 해결하는 방안을 소개하는 논문이지만 해결 방식에서 positional encoding을 따로 적용해주는 방식을 쓰고 있다([자세한 설명은 여기](https://github.com/bcaitech1/p4-fr-9-googoo/discussions/9)).
 - 해당 논문에서 제시한 방법을 사용하면 positional encoding에 대한 query,key의 내적을 따로 미리 계산해 두고 positional encoding이 더해지지 않은 feature map을 channel attention으로 먼저 통과시켜 주게 된다. 그 다음 self attention 단계에서는 channel attention의 output(여기에는 feature map의 정보만 있다. positional encoding에 대한 정보는 없음)을 이용하여 query, key ,value를 만들고 query, key의 내적를 계산한다. 그 후 여기에 미리 계산해놓은 positional encoding에 대한 query,key의 내적을 더해준 후 softmax를 통과시켜주면 attention weight 계산이 완료되는 것이다. 아래 그림에서 왼쪽이 기존 방식이고 오른쪽이 논문에서 제시한 방식을 응용한 것이다.

![image](https://user-images.githubusercontent.com/51700219/122532969-025ab600-d05c-11eb-986c-f18489242d29.png)
 - 이런식으로 channel attention을 적용하는 단계에서 positional encoding의 영향을 받지 않게 할 수 있다.

![image](https://user-images.githubusercontent.com/51700219/129671053-492a845e-62ee-4beb-9aa0-56ab71898f97.png)
 - positional encoding을 따로 적용한 상태에서 channel attention을 적용함으로써 accuracy를 올릴 수 있었다(SATRN: 기존모델, SATRN_channel_attn_PE: channel attention을 적용하고 positional encoding을 따로 적용한 모델).

 - 근데 점수가 올라간 게 channel attn을 적용한 것 때문이 아니라 positional encoding을 따로 적용해서 올라간게 아닐까 하는 생각이 들어서 따로 더 실험을 해보았다.
 - 기존 모델에 channel attention은 적용하지 않은 상태로 positional encoding을 따로 적용한 것도 실험해 보았다(SATRN: 기존모델, SATRN_PE: positional encoding을 따로 적용한 모델).

![image](https://user-images.githubusercontent.com/51700219/129670884-94984116-175f-42d4-915b-bd3dccc804be.png)
 - positional encoding을 따로 적용해도 점수가 오른다. 다만 여기에 channel attention을 적용해 주면 더 올릴 수 있다.
### 6/
 - 학습이 완료된 모델로 validation dataset에 대해서 inference를 한 후 잘 맞추는 데이터와 못 맞추는 데이터를 나눠서 살펴보았다.
 - 대체적으로 잘 맞추는 데이터는 sequence 길이가 짧고 잘 못 맞추는 것은 sequence 길이가 긴 편이다. 못맞춘 데이터들 중에 sequence가 짧은 데이터의 경우 같은 토큰을 연속으로 예측해서 아쉽게 틀리는 경우와 대소문자를 잘못 구분하거나 비슷한 다른문자로 인식하는 경우(근데 직접 눈으로 봐도 헷갈리만 하다)도 많았다.
 - 아래 그림은 맞춘 데이터(correct)와 틀린 데이터(error)의 길이 분포이다. x축은 sequence length이고 y축은 데이터 갯수이다.

![image](https://user-images.githubusercontent.com/51700219/120808342-f2cf6d80-c583-11eb-811e-72cce42f8480.png)
 - 긴 sequence일수록 모든 time step에서 정확한 토큰을 예측하여 정확히 문장전체를 예측하기가 어려워 진다(문장 전체를 정확히 예측할 확률인 joint probability자체가 낮아질 수 밖에 없음).
 - 일단 beam search를 이용하면 greedy보다는 확률이 높은 sequence를 찾을 수 있기 때문에 긴 문장을 예측하는데 도움이 될 것 같다. 또한 같은 토큰을 연속해서 예측한 결과 뿐만 아니라 다른 예측결과도 찾게 되므로 같은 토큰을 연속으로 예측하는 경우도 줄여줄 수 있을 것 같다.
 - 시간이 부족하여 구현하지 못하였다.



### 아쉬운 점
 - teacher forcing ratio를 조금씩 낮추는 방식을 실험해보고 싶었는데 못했다
 - 1등팀은 teacher forcing ratio를 cos형태로 낮추는 방식을 사용하였는데 점수가 크게 올랐다고 한다. 일반화성능을 높이는데 좋은 기술인 것 같다.
 - 90도, 180도, 270도 돌아간 데이터들에 대해서 주어진 이미지가 몇도 돌아간 이미지인지 분류를 해주는 모델을 학습해보지 못한게 아쉽다. 이러한 간단한 분류기로 90도, 180도, 270도 돌아간 이미지같은 아웃라이어를 커버해줄 수 있었을 것 같다. 
