GAN과 오토인코더를 조합해서 robust한 특성을 추출하고 fingerprint 특성만 조작함.
Fh: Design Principles of Convolutional Neural Networks for Multimedia Forensics
Fl: 가우시안
E: Design Principles of Convolutional Neural Networks for Multimedia Forensics
G: MUNIT
D: PatchGAN

# Abstract

멀티미디어 기술과 공유 플랫폼의 발전으로 인하여 사실적인 이미지를 합성하는 것이 매우 쉬워졌다. 이러한 이미지는 사회적 혼란을 야기할 수 있는 위험성이 높은 잘못된 정보를 전달할 수 있다. 따라서 조작된 멀티 미디어를 구별할 필요가 있으며 분석할 수 있는 탐지기술에 대한 관심이 빠르게 증가하고 있다. 이미지 포렌식 분야에서 사용되는 대부분의 탐지 기법들은 이미지가 생성 및 촬영된 기기 혹은 소프트웨어가 남기는 작은 흔적에 의존하여 작동된다. 특히, 카메라의 경우 이미지를 생성하는 과정에서 센서의 물리적 특성으로 인하여 각기 다른 구분가능한 흔적을 남긴다. 본 논문에서는 GAN과 Autoencoder를 활용하여 기존의 탐지 기법을 무시할 수 있도록 흔적을 추출 및 합성하고, 합성하는 과정에서 추출된 구분가능한 특성을 활용하여 보다 강인한 탐지를 가능하게 한다.

# Introduction
스마트폰 대중화와 SNS 이용의 일상화는 우리 사회에 많은 변화를 가져다 주었고, 이전과 비교하여 매우 쉽게 이미지를 촬영하고 공유할 수 있게 되었다. 이로인해 프라이버시 침해, 사이버폭력, 허위정보의 확산등 여러 부작용을 초래하고 있으며 여러가지 디지털 이미지 포렌식 기법들이 주목받고 있다. 여러 디지털 이미지 포렌식 기법들은 
촬영 기기를 식별하는 포렌식 기법 
본 논문에서는 통해 보다 강인한 특성 추출

# Related Works

카메라 모델 식별의 발전사, CFA, PipeLine
대표적인 CMI 논문 하나 넣고 설명

사람재인식 강인한 특성이 필요함 CMI또한 컨텐츠 특성에 영향을 받지 않는 지문 특성을 뽑아내야함
DG-NET 설명, 인코더가 카메라 지문 특성만을 GAN을통해 유도

## Papers

### Design Principles of Convolutional Neural Networks for Multimedia Forensics

#### Introduction

Multimedia information, such as digital images, is frequently used in numerous important settings, such as evidence in legal proceedings, criminal investigations, and military and defense scenarios.
디지털 이미지와 같은 멀티 미디어는 legal proceedings, criminal investigations, and military and defense scenarios와 같은 수많은 환경에서 주로 사용된다.
Since this information can easily be edited or falsified, information forensics researchers have developed a wide variety of methods to forensically determine the authenticity and source of multimedia data [27].
Many early forensic approaches were developed by theoretically or heuristically identifying a set of traces left in an image by a particular processing operation or source device.
For example, techniques have been developed to detect specific traces left by resampling and resizing [22, 14], contrast enhancement [26], median filtering [15, 13], sharpening [2], and many other operations. Similarly, forensic algorithms have been developed to identify the model of an image’s source camera using specific traces left by different elements of the camera’s internal processing pipeline [28, 3, 8]

### Remnant Convolutional Neural Network for Camera Model Identification - RemNet

#### Introduction

Camera model identification (CMI) has gained significant momentum in recent years for information forensics as digitally altered images are becoming more pervasive in electronic media [1].

The increased usage of digital images in our everyday-life for entertainment, social networking, and more importantly in legal and security issues is, therefore, raising authenticity concern regarding the source of an image and its content, especially when presented to a court as an evidence [2].
Furthermore, the available professional image editing tools, though intended for entertainment purposes, are also facilitating image manipulation for illegal acts, making the problem of CMI even crucial.
Although the metadata of an image contains some information about the source, it is not a reliable metric to determine the source since this data can be forged [1].
Besides, the metadata of the digital images are mostly arXiv:1902.00694v3 [eess.IV] 27 Jun 2020 2 Abdul Muntakim Rafi1 et al. unavailable when shared online in social media.
Moreover, while sharing online, images go through various post-processing operations which destroy the trace of the source information to some extent, making the identification even more difficult [3].
As a result, the task of identifying the camera model is continually becoming more challenging.
Therefore, a forensic analyst has to resort to image processing and analysis techniques to identify the camera model with which an image was taken.
