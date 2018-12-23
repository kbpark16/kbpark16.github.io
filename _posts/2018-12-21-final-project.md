---

title: "Semi-supervised Learning-Generative Models"

작성자: KeonVin-Park
use_math: true

---

본 포스팅에서는 Semi-supervised Learning 방법 중 하나인 Generative Models, 그 중에서도 Gaussian mixture model 에 대해 자세히 다루겠습니다. 글의 전체적인 내용은 고려대학교 강필성 교수님의 Business-Analytics 강의를 참고하였음을 밝힙니다.

_ _ _

**Semi-supervised Learning의 개념**

_ _ _

Generative model에 대해 설명하기에 앞서, Semi-supervised Learning이 뭔지에 대해 간략히 설명드리도록 하겠습니다.
Machine Learning은 크게 Regression, Classifcation과 같이 Label이 있는 Supervised Learning과, Clustering과 같이 Label이 없는 Unsupervised Learning으로 나뉩니다. 그 중에서도 Semi-Supervsied Learning은 Supervised Learning의 성능을 높이기 위해서, Label이 없는 Unlabeled data까지 활용하는 학습방법을 의미합니다. 이러한 Semi-Supervsied Learning은 labeled data의 '절대적인' 갯수가 적을때  높은 performance 향상을 기대 할 수 있습니다. 예를 들어, 전체 데이터 중에서 labeled data의 비율이 1%(unlabeled data는 99%)인데 100만개일 때보다, labeled data의 비율이 99%(unlabeled data는 1%)여도 50개 일 때 더 높은 성능 향상을 기대 할 수 있습니다.
아래는 전체적인 이해에 도움이 되는, 강필성 교수님의 교안에서 얻은 그림입니다.

![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/semi-supervised-learning.PNG?raw=true)

reference: 강필성 교수님 '비즈니스 어낼리틱스' 교안 chapter5, p.3

_ _ _

**Self-Training **

_ _ _

뒤에서 설명드릴 Semi-supervised Learning으로 쓰이는 Gaussian mixture model은 아래와 같이 Self-training의 한 방식입니다.
간단하게 말씀드리면, 시작은 labeled data만 이용해서 모델을 학습시키고, 그 모델을 바탕으로 unlabeled data의 label을 예측합니다.
그리고 그 예측결과를 바탕으로 unlabeled data에 label을 주고, 기존 labeled data와 합쳐서 새로 모델을 학습시키는 것입니다.
뒤에서 말씀드리겠지만, Gaussian mixture model은 \(\omega,\mu,\sigma\) 3개 파라미터가 있고, 파라미터가 특정 값에 수렴할때까지 1,2,3번 과정을 반복하는 EM algorithm을 이용합니다.

![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/self_training.PNG?raw=true)
reference: https://www.pinterest.co.kr/pin/540713499003163569/,  강필성 교수님 '비즈니스 어낼리틱스' 교안 chapter5, p.19

_ _ _

_ _ _

**Discriminative model v.s Generative model **
_ _ _
Generative model 중 하나인 Gaussian mixture model을 설명하기에 앞서, 아래 그림으로 Generative model과 Discriminative model의 차이를 설명드리겠습니다. 아래에서 h는 머신러닝에서 y(종속변수),v는 x(예측변수)라고 생각하시며 됩니다. 흔히 아시는 Discriminative model은, x가 주어졌을때 y의 확률을( p(y|x) ) 구해서 classification을 하는 모델로써, 대표적으로 로지스틱 회귀분석을 생각하시면 됩니다. Generative model은 똑같이 classification이 목적이지만, 이 데이터가 어떠한 메커니즘( p(x,y) )에 의해 형성되었는지를 알고, 그것을 기준으로 classification을 하는 방식입니다.  (p(x,y)=p(x|y)p(y): Using Bayes' theorem)
![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/Dis-Gen.PNG?raw=true)
reference: Choi, S. (2015). Deep Learning: A Quick Overview. Learning Choi, S. (2015). Deep Learning: A Quick Overview,  강필성 교수님 '비즈니스 어낼리틱스' 교안 chapter5, p.28

_ _ _

**GMM for Novelty Detection v.s GMM for Semi-supervised Learning **

_ _ _


Semi-supervised Learning: Generative models, especially Gaussian mixture models, Novelty detection과 차이: 둘 다 궁극적으로는 classification(Supervised Learning)의 performance를 높이고자 하는 시도인데, Novelty detection에서의 GMM은 BINARY로 치면, 불량/양품중에 양품이 절대적으로 수가 적을 때, 양품으로만 \(\omega,\mu,\sigma\) 파라미터를 학습시키고, 그 결과로 불량품을 분류하는데, Semi-supervised Learning은 labeled data의 수가 절대적으로 부족할 때,  불량과 양품 모두 학습시키는 대신에, 시작은 labeled data로만 이용해서 파라미터를 학습시키고, 그것을 바탕으로 unlabeled data도 classification 후, 학습 데이터로 추가시키는 self-training 종류의 하나입니다. 결과적으로 둘 다 Classification을 잘 하기 위한 것이고 같은 모델(GMM)을 사용하지만, 가장 큰 차이는 양품만 이용하느냐, 불량품 양품 모두 이용하는데 unlabeled data까지 이용하느냐 차이입니다.


_ _ _

**GMM for semi-supervised learning**

_ _ _
unlabeled data까지 활용하는 Semi-supervised learning에 앞서,
아래와 같이 labeled data만 있을때의 GMM을 이용한 classification(Supervised Learning)에 대해서 설명하겠습니다.
labeled data만 있는 경우(group=1,2) 아래와 같이 모델 파라미터에 대한 MLE를 명시적으로 구할 수 있으므로 쉬운 문제입니다.
![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/labeled_data_only.PNG?raw=true)
reference: Zhu.X.(2007). Semi-Supervised Learning Tutorial,  강필성 교수님 '비즈니스 어낼리틱스' 교안 chapter5, p.29
![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/MLE.png?raw=true)
and  $$(\omega_{1}: proportion of class 1, \omega{2}: proportion of class 2\)$$

reference: CS229 Lecture notes,Andrew Ng, Part IV, Generative Learning algorithms

위와 같이 Labeled data만 있는 경우에는, 모든 파라미터에 대한 MLE를 명시적으로 구할 수 있기 때문에 MLE에 근사한 해를 찾기 위한 EM 알고리즘 등 의 시도가 필요하지 않습니다.

하지만, 아래와 같이 unlabeled data까지 고려하는 경우(Labeled and unlabeled data), unlabeled data의 실제 label을 모르기 때문에, unlabled data의 label이 hidden variable이 되고, 아래의 Log-likelihood function를 Maximize 시켜주는 MLE를 명시적으로 구하기 어렵습니다. 그러므로, MLE를 근사적으로 구하는, Expectation-Maximization(EM)algorithm을 이용하여 parameter\(\theta={\omega,\mu,\sigma}\)들의 최적값을 구해야 합니다.

![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/unlabeled_data_added.PNG?raw=true)
reference: Zhu.X.(2007). Semi-Supervised Learning Tutorial,  강필성 교수님 '비즈니스 어낼리틱스' 교안 chapter5, p.34
위에서 언급한대로, unlabeled data가 있는 경우에, E-M 알고리즘을 통해 M.L.E에 근사적인 해를 구하는 과정은 아래와 같습니다.
Step0(Initialization step)에서 labeled data만 이용하여 각 클래스별(group=1,2) 비율, 모평균벡터, 모공분산행렬의 M.L.E를 
초기값으로 구합니다. 비율은 단순히 클래스별 비율로 추정하고, 모평균벡터 및 모공분산행렬의 M.L.E는 Andrew Ng님의 reference로 위에서 명시하였습니다.
Step1(Expectation step)에서는 모든 unlabeled data에 대해, \(\p(y|x,\theta))\를 계산하여 더 높은 확률을 갖는 클래스로 할당합니다.
-수식 풀어서 wy 보여주기
Step2(Maximization step)에서는 labeled data에 Step1에서 \(\p(y|x,\theta))\를 기반으로 labeling 한 unlabeled data들을 학습데이터로 추가하여, proportion, sample mean, sample covariance로 각 클래스별 \(\theta_{MLE}\)를 업데이트 합니다.
그리고 Step1, Step2를 theta가 특정 값으로 수렴할때까지 반복합니다.
-unlabeled data에 대해서만 Step1, Step2를 반복하는 것인가?
-iteration 은 하이퍼 파라미터? 값들이 수렴하는지 보고

![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/E-M%20algorithm.PNG?raw=true)
reference: Zhu.X.(2007). Semi-Supervised Learning Tutorial,  강필성 교수님 '비즈니스 어낼리틱스' 교안 chapter5, p.35

where,
$$P(x,y | \theta)= P(y | \theta)P(x | y , \theta)=\omega_{y}N(x; \mu_{y},\Sigma_{y})$$

$$\sum_{y'}P(x,y'|\theta)=P(y=1|\theta)P(x|y=1,\theta)+P(y=2|\theta)P(x|y=2,\theta)$$

with assumption that(GMM),
$$ x|y=1 follows MN(\mu_{1},\Sigma_{1})$$

$$ x|y=2 follows MN(\mu_{2},\Sigma_{2})$$

$$ y follows Bernoulli(p=P(Y=2)), y=1,2$$

1. Expectation stage


2. Maximization stage

#(Gaussian mixture model-EM algorithm)
```ruby
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt    

# 이변량 정규분포(X1,X2)에서 각각 모평균벡터, 모공분산 행렬을 다르게 갖는
#  2가지의 분포에서 500개씩의 데이터 추출(난수 생성)
#(총 데이터 갯수=1000 (labeled data+ unlabeled data)

#group1의 모평균, 모 공분산행렬 정의 및 랜덤 시드 지정 후 500개 난수 추출
mean1=[3,1]
sigma1=[[1,0],[0,1]]
np.random.seed(0)
N1=np.random.multivariate_normal(mean1,sigma1,500).T

#group2의 모평균, 모 공분산행렬 정의 및 랜덤 시드 지정 후 500개 난수 추출
mean2=[2,2]
sigma2=[[1,0.9],[0.9,1]]
np.random.seed(1)
N2=np.random.multivariate_normal(mean2,sigma2,500).T

# 위에서 뽑은 1000개의 데이터 중에서(group1:500,group2:500)
#labeled data 생성-각각 50개씩만-semi-supervised learning은
#labeled data의 절대적인 수가 적을때 performacne 향상이 좋으므로
labeldata1={}
#python은 index가 0부터 시작하므로 50까지 주면 0부터 49번째 index,즉 50개가 뽑힙니다.
for i in range(50):
	labeldata1[N1[0][i],N1[1][i]]='blue' #N1[0][i]: group1의 X1, N1[1][i]: group1의 X2
#group1의 범주를 blue로, group2의 범주를 red로 줌 
labeldata2={}
for i in range(50):
	labeldata2[N2[0][i],N2[1][i]]='red' #N2[0][i]: group2의 X1, N1[1][i]: group2의 X2

# unlabeled data 생성
non_labeldata={}
#갯수: 50번째 index 데이터부터 나머지까지(500): unlabeled data: 450+450=900게
for i in range(50,len(N1[0])):
	non_labeldata[N1[0][i],N1[1][i]]='' 
	non_labeldata[N2[0][i],N2[1][i]]=''

# 시각화를 위한 label, unlabeled data 키값 축출
labeldata1_key=list(labeldata1.keys())
#여기서의 x는 Group1의 X1, y는 Group1의 X2를 의미합니다.(label이 아닙니다!)
labeldata1_x=[]
labeldata1_y=[]
#여기서의 x는 Group2의 X1, y는 Group2의 X2를 의미합니다.(label이 아닙니다!)
labeldata2_key=list(labeldata2.keys())
labeldata2_x=[]
labeldata2_y=[]
#여기서의 x는 Group1과 Group2를 합친, 
#non-labeled data의 X1, y는 non-labeled data의 X2를 의미합니다.(label이 아닙니다!)
non_labeldata_key=list(non_labeldata.keys())
non_labeldata_x=[]
non_labeldata_y=[]

for i in range(len(labeldata1_key)):
	labeldata1_x.append(labeldata1_key[i][0])
	labeldata1_y.append(labeldata1_key[i][1])
	labeldata2_x.append(labeldata2_key[i][0])
	labeldata2_y.append(labeldata2_key[i][1])

for i in range(len(non_labeldata_key)):
	non_labeldata_x.append(non_labeldata_key[i][0])
	non_labeldata_y.append(non_labeldata_key[i][1])

# 후에 EM-algorith에 필요
total_x1=labeldata1_x+non_labeldata_x
total_y1=labeldata1_y+non_labeldata_y
total_x2=labeldata2_x+non_labeldata_x
total_y2=labeldata2_y+non_labeldata_y
    
    
# labeled:group1,group2, unlabeled data 시각화에 필요한 패키지(X축:X1,Y축:X2)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# labeleled(group1:blue, group2: red, unlabeled data 시각화)

plt.xlabel('X1') #xlabel: X1-axis
plt.ylabel('X2') #ylabel: X2-axis
plt.title('Visualizing Distribution of group1,group2,unlabeled data')

plt.scatter(non_labeldata_x,non_labeldata_y,c="y",s=1) #yellow: unlabeled data
plt.scatter(labeldata1_x,labeldata1_y,c="b",s=15) #group1: blue
plt.scatter(labeldata2_x,labeldata2_y,c="r",s=15) #group2: red

blue_patch = mpatches.Patch(color='blue', label='blue(group1)')
red_patch = mpatches.Patch(color='red', label='red(group2)')
yellow_patch = mpatches.Patch(color='yellow', label='yellow(unlabeled data)')

plt.legend(handles=[blue_patch,red_patch,yellow_patch])

plt.show()

```
![](https://github.com/kbpark16/kbpark16.github.io/blob/master/images/new_dist.PNG?raw=true)


```ruby
```
![](EM-algorithm 학습 과정(수렴할때까지 iteration)-반복결과)

#모평균벡터/모평균 공분산행렬 보여주고, 위에서 얻은 값과 얼마나 차이나는지, EM-algorithm을 이용한 Semi-supervised learning의 수치적 결과와 labeled data만 이용한 Supervised learning의 수치적 결과를 제시하고, 둘의 결과 plotting(분포 및 decision boundary)
그래서 결론적으로 50개,50개 100개 labled data: unlabled data 몇게, 100개 있으니까 절대적으로 작아서 performance가 좋았다. 가 결론이 되야함.
-시각화가 매우 중요!
```ruby
print("----------------------- ","exact solution"," -----------------------",sep="")
print(mean1,"|",mean2)
print(sigma1)
print(sigma2)
```


#limitation: MN로뿌터 뽑은 simulation data라서 GMM알고리즘이 잘작동되었을 수 있고, 일반적으로 얻을 수 있는 데이터는 MN 가정을 만족하지 않을 수 있으므로 performance는 안 좋을수 있다.-안 좋았던 수업 ppt 예 보여주기(시간남으면 MN이랑 따른 분포로부터 뽑은 데이터랑 performance 비교 )
