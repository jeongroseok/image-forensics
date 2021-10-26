### 7. 안티 포렌식 & 카운터 안티 포렌식
이미지 안티 포렌식은 이미지를 수정하여 포렌식 알고리즘이 작동되지 않도록 하는 기법. 이러한 연구를 하는 이유는 포렌식 커뮤니티의 발전을 위함

최근 gan을 활용하는 쪽으로 가고 있음
[102] 논문에서 DenseNet-40을 카메라 출처 확인을 위해 학습하고, FGSM과 JSMA 공격에 취약하단걸 확인함.

[105] 논문에선 여러 모델을 patch와 full res 입력에서 두가지 공격법을 수행해보고 공부함.

#### 발표 내용

적대적 공격 이란?
- 적대적 공격의 목표: 신뢰도 저하, 오분류, 의도된 오분류, 원본/목표 오분류(의도된 오분류는 모든 입력에대해 출력이 1개; 원본/목표 오분류는 특정 입력이 특정 목표로 보이게 만듦.)
적대적 사례 생성: 의도적으로 오분류되는 입력을 만드는 것
적대적 학습: 적대적 사례를 집어넣은 데이터로 학습 시키는 것
Defensive Distillation: 원래 모델을 모방하는 두번째 모델을 학습 하는것
FGSM 설명
블랙박스와 화이트박스
오분류와 소스/타겟 오분류

#### 메모
적대적 사례 생성이 중요한 이유?
인쇄를 해도, 스캔을 해도 통함
비슷한 데이터로 훈련됐을경우에도 범용적으로 통할 수 있음
화이트박스 공격시 모델이 필요한데, 대상 모델을 모방하도록 훈련도 가능함. 화이트박스를 미러링을 통해 블랙 박스 공격이 가능해짐.

적대적 훈련시 비슷한 유형의 공격에 저항성을 가짐.
https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-8-d9507cf20352

FGSM 식 설명:
https://aistudy9314.tistory.com/37
https://velog.io/@miai0112/Explaining-And-Harnessing-Adversarial-Examples-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
https://noru-jumping-in-the-mountains.tistory.com/16?category=1218655
https://rain-bow.tistory.com/entry/%EC%A0%81%EB%8C%80%EC%A0%81-%EA%B3%B5%EA%B2%A9Adversarial-Attack-FGSMPGD

Camera identification with deep convolutional networks, 3 Mar 2016
