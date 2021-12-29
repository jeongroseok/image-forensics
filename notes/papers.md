# Design Principles of Convolutional Neural Networks for Multimedia Forensics
이미지 포렌식 분야에서 CNN 아키텍쳐를 디자인하는 가이드라인 제시
## datasets
IFS-TC 데이터셋 사용
1024 x 768이미지를 256 x 256 패치로 나누고 중앙의 9개를 사용함
we divided these images into 256×256 pixel patches and retained all the nine central patches from the green layer of each image for our training database.

More specifically, the following fixed HPF has been used in steganography [21]:
This method has proven effective and promising at distriguishing between the stego and cover images.
이 방법은 스테고와 커버 이미지 사이의 분리에 효과적이고 유망한 것으로 입증되었다.

# Deep Learning for steganalysis is better than a Rich Model with an Ensemble Classifier, and is natively robust to the cover source-mismatch
We observed that CNNs do not converge without this preliminary high-pass filtering.
CNN이 이 예비 하이패스 필터링 없이는 수렴되지 않는다는 것을 관찰했다.