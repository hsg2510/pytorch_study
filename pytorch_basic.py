'''
    1. torch 
    메인 네임스페이스. 텐서 등의 다양한 수학 함수가 포함되어져 있으며 Numpy와 유사한 구조를 가진다. 

    2. torch.autograd 
    자동 미분을 위한 함수들이 포함되어져 있다. 자동 미분의 on/off를 제어하는 콘텍스트 매니저(enable_grad/no_grad)나 자체 미분 가능 함수를 정의할 때 사용하는 기반 클래스인 'Function' 등이 포함되어져 있다.

    3. torch.nn
    신경망을 구축하기 위한 다양한 데이터 구조나 레이어 등이 정의되어져 있다. 예를 들어 RNN, LSTM과 같은 레이어, ReLU와 같은 활성화 함수, MSELoss와 같은 손실 함수들이 있다.

    4. torch.optim
    확률적 경사 하강법(Stochastic Gradient Descent, SGD)를 중심으로 한 파라미터 최적화 알고리즘이 구현되어져 있다. 

    5. torch.uitls.data
    SGD의 반복 연산을 실행할 때 사용하는 미니 배치용 유틸리티 함수가 포함되어져 있다. 

    6. torch.onnx
    ONNX(Open Neural Network Exchange)의 포맷으로 모델을 export할 때 사용한다. ONNX는 서로 다른 딥 러닝 프레임워크 간에 모델을 공유할 때 사용하는 포맷이다.
'''

import numpy as np

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print("Rank of t", t.ndim) 
print("Shape of t", t.shape) # shape은 크기를 의미. (7, ) = (1, 7)을 의미


##### pytorch #####
import torch 

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t.dim())
print(t.shape)
print(t.size())

t = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])

print(t.size())

## Broadcasting
# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) 
print(m1 + m2)

# Vector(2 * 1) * Vector(1 * 2)
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2) # [[4, 5], [5, 6]] -> m1, m2 모두 2 * 2 행렬로 확장 됨.

# Matrix Multiplication VS Multiplication
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2)) # 행렬 곱셈
print(m1 * m2) # 원소 곱셈 -> braodcasting 후 같은 위치의 원소끼리 곱셈

# Mean
print(m1.mean()) # 모든 원소의 평균, size 1의 1차원 텐서가 됨.

# dim=0에서 0은 행을 뜻함.dim=0 이라는 뜻은 행을 제거한다 라는 뜻. 즉, 열만 남기므로 결과적으로는 같은 열끼리 더해져서 평균을 내게 됨. sum, max, min도 마찬가지.
print(m1.mean(dim=0)) #[(1+3)/1, (2+4)/2]의 size 2의 1차원 텐서가 됨. 

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.max(dim=0)) # torch.return_types.max 라는 객체를 리턴하는데 values, indices property가 있음. values는 최대값 tensor, indices는 최대값의 인덱스 tensor를 return
print(t.max(dim=0).values)
print(t.max(dim=0).indices)

# view 
t = np.array([
    [[0, 1, 2], 
     [3, 4, 5]], 
    [[6, 7, 8], 
     [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape) # 2 * 2 * 3
print(ft.view([-1, 3])) # -1은 첫번째 차원은 사용자가 모르겠으니 파이토치에게 맡기겠다는 의미. 2*2*3 -> 4*3

# squeeze -> 1인 차원을 제거, unsqueeze -> 특정 위치에 1인 차원을 추가
ft = torch.FloatTensor([[0], [1], [2]])
print(ft.squeeze()) # 1차원 제거 [0, 1, 2] 

ft = torch.Tensor([0, 1, 2])
print(ft.unsqueeze(0)) # (3, ) 크기를 가졌던 1차원 벡터가 (1, 3)인 2차원 벡터로 변경됨. -> [[0, 1, 2]]
print(ft.unsqueeze(1)) # (3, ) 크기를 가졌던 1차원 벡터가 (3, 1)인 2차원 벡터로 변경됨. -> [[0], [1], [2]]

# Type Casting
'''
Data Type
- torch.FloatTensor() : torch.float32 or torch.float -> 32-bit float <-> torch.cuda.FloatTensor()
- torch.DoubleTensor() : torch.float64 or torch.double -> 64-bit float <-> torch.cuda.DoubleTensor()
- torch.HalfTensor() : torch.float16 or torch.half -> 16-bit float <-> torch.cuda.HalfTensor()
- torch.IntTensor() : torch.int32 or torch.int -> 32-bit int (signed) <-> torch.cuda.IntTensor()
- torch.LongTensor() : torch.int64 or torch.long -> 64-bit int (signed) <-> torch.cuda.LongTensor()
- torch.ShortTensor() : torch.int16 or torch.short -> 16-bit int (signed) <-> torch.cuda.ShortTensor()
- torch.ByteTensor() : torch.uint8 -> 8-bit int (unsigned) <-> torch.cuda.ByteTensor()
- torch.CharTensor() : torch.int8 -> 8-bit int (signed) <-> torch.cuda.CharTensor()
''' 
lt = torch.LongTensor([1, 2, 3, 4])
print(lt.float()) # float로 변환

# Concatenate
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0)) # 어느 차원으로 늘릴지 dim으로 지정 가능 -> 4 * 2 행렬이 됨.

# Stacking
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z])) # 3 * 2 행렬이 됨. 기본적으로 dim=0으로 쌓음. dim=1로 하면 2 * 3 행렬이 됨. 즉 dim이 증가하도록 쌓으라는 뜻.

# Ones and Zeros like
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(torch.ones_like(x)) # x와 같은 크기의 1로 이루어진 텐서
print(torch.zeros_like(x)) # x와 같은 크기의 0으로 이루어진 텐서

# In-place Operation
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
print(x) # x는 변화가 없음.

print(x.mul_(2.)) # 곱하기 2를 수행한 결과를 출력
print(x) # x도 변화됨.