이 논문에서 제안하는 방법론은 **BER (Bi-directional Energy Regularization)**입니다. BER은 Class Incremental Learning(CIL) 환경에서 발생하는 두 가지 주요 편향(새로운 클래스에 대한 편향, 오래된 클래스에 대한 망각)을 완화하여 OOD(Out-of-Distribution) 탐지 성능을 높이기 위해 고안되었습니다.

구현을 위해 논문에 기술된 BER의 상세 구조와 수식을 단계별로 설명해 드립니다.
1. 전체 프레임워크 및 설정
BER은 Fine-tuning 기반의 OOD 탐지 프레임워크를 따릅니다.

모델 구조: 각 증분 단계 $t$에서 CIL 모델 $\theta_t(\cdot)$는 특징 추출기(Feature Extractor) $\phi_t(\cdot)$와 분류기(Classifier) $h_t(\cdot)$로 구성됩니다.


학습 전략:
기존 CIL 모델의 특징 추출기 $\phi_t(\cdot)$와 분류기 $h_t(\cdot)$는 **동결(Freeze)**하여, 원래의 ID(In-Distribution) 분류 성능에 영향을 주지 않도록 합니다.


OOD 탐지를 위해 추가적인 분류기(Extra Classifier) $f_t(\cdot)$만을 특징 추출기 $\phi_t(\cdot)$ 위에 추가하여 Fine-tuning합니다.


이 $f_t(\cdot)$는 현재 단계의 학습 데이터 $T_t^{train}$ (새로운 태스크 데이터 + 메모리에 저장된 과거 데이터)만을 사용하여 학습됩니다.


2. NTER (New Task Energy Regularization)
NTER은 모델이 OOD 샘플을 새로운 태스크의 클래스로 오분류하는 경향을 줄이기 위해 설계되었습니다. 이를 위해 새로운 클래스 샘플들을 섞어(Mixup) 가짜 OOD(Pseudo-OOD)를 생성하고 결정 경계(Decision Boundary)를 조절합니다.

Pseudo-OOD 생성: 새로운 태스크의 훈련 배치 $X_t$에서 샘플 $x_t^i$와 $x_t^j$ ($y_t^i \ne y_t^j$)를 선택하여 다음과 같이 Pseudo-OOD 샘플 $\tilde{x}_t$를 합성합니다.

$$\tilde{x}_{t}=\beta x_{t}^{i}+(1-\beta)x_{t}^{j}$$
여기서 $\beta \in [0, 1]$은 Beta 분포에서 샘플링된 값입니다. 효율성을 위해 전체 데이터셋의 조합 대신 미니 배치 내에서 셔플링을 통해 쌍을 생성합니다.


NTER 손실 함수 ($\mathcal{L}_n$): 에너지 기반 손실 함수를 사용하여, 실제 새로운 클래스 데이터($x_t$)의 에너지는 낮추고($p_{in}$), 합성된 Pseudo-OOD 데이터($\tilde{x}_t$)의 에너지는 높입니다($p_{out}$).

$$\mathcal{L}_{n}=\mathbb{E}_{x_{t}\sim X_{t}^{train}}[(max(0, E(x_{t}) - p_{in}))^{2}]+\mathbb{E}_{\tilde{x}_{t}}[(max(0, p_{out} - E(\tilde{x}_{t})))^{2}]$$
여기서 에너지 함수 $E(x; f)$는 다음과 같이 정의됩니다[cite: 1346]. $$E(x;f)=-\tau\cdot log(\sum_{j=1}^{Q_{t}}e^{f_{j}(x)/\tau})$$ ($\tau$는 온도 스케일링 하이퍼파라미터, $Q_t$는 현재 단계까지의 전체 클래스 집합입니다.)
3. OTER (Old Task Energy Regularization)
OTER은 오래된 태스크의 클래스에 대한 낮은 예측 신뢰도 문제를 해결하여, 과거 클래스 샘플이 OOD로 오분류되는 것을 방지합니다.

합성된 과거 클래스 샘플 생성 (Synthesized Old Sample): 메모리 버퍼에 저장된 적은 수의 과거 클래스 샘플 $M_t$의 다양성을 높이고 정보를 전이하기 위해, 새로운 클래스 샘플 $X_t$와 섞습니다(Mixup).

$$\overline{m}_{t}=\lambda x_{t}+(1-\lambda)m_{t}$$
여기서 $\lambda$는 Mixup 하이퍼파라미터, $m_t$는 메모리의 과거 클래스 샘플입니다.


OTER 손실 함수 ($\mathcal{L}_o$): 합성된 과거 클래스 샘플($\overline{m}_t$)의 에너지를 낮추어 ID 데이터로 확실히 인식되도록 합니다. 주의할 점은, NTER과 달리 OTER에서는 Pseudo-OOD에 대한 에너지 정규화 항을 사용하지 않습니다. (과거 샘플로 Pseudo-OOD를 만들면 결정 경계가 지나치게 압축될 수 있기 때문입니다 ).

$$\mathcal{L}_{o}=\mathbb{E}_{(x_{t},m_{t})\sim(X_{t}^{train},M_{t})}[(max(0, E(\overline{m}_{t})-p_{in}))^{2}]$$
4. 최종 최적화 목표 (Optimization Objective)
추가 분류기 $f_t(\cdot)$를 학습하기 위한 최종 손실 함수는 Cross-Entropy 손실과 두 가지 **에너지 정규화 손실(NTER, OTER)**의 합으로 구성됩니다.


$$\mathcal{L}=\mathbb{E}_{(x,y)\sim(T_{t}^{train},Q_{t})}[\mathcal{l}(f(x),y)]+\alpha(\mathcal{L}_{n}+\mathcal{L}_{o})$$
$\mathcal{l}(\cdot)$: Cross-Entropy Loss


$\alpha$: 하이퍼파라미터 (논문에서는 0.1 등 실험 설정에 따름)


$\mathcal{L}_n$: 식 (3)의 NTER 손실
$\mathcal{L}_o$: 식 (5)의 OTER 손실
5. 추론 (Inference)
학습이 완료된 후, 테스트 단계에서는 Fine-tuned된 분류기 $f_t(\cdot)$를 사용하여 입력 데이터의 **에너지 점수(Energy Score)**를 계산하고, 이를 OOD 점수로 사용합니다.


$$Energy\ Score(x) = -\tau\cdot log(\sum_{j=1}^{Q_{t}}e^{f_{j}(x)/\tau})$$
(참고: $f_t$는 OOD 탐지용이며, 실제 클래스 분류(ID Classification)가 필요한 경우 원래의 $h_t$를 사용합니다.)

