X, y = zip(['a', 1], ['b', 2], ['c', 3])
print(X)
print(y)

# 지금은 list의 list지만, 행렬 or 2d tensor도 가능 
sequences = [['a', 1], ['b', 2], ['c', 3]]
X, y = zip(*sequences)
print(X)
print(y)

# DataFrame 분할하는 방법
import pandas as pd

values = [['당신에게 드리는 마지막 혜택!', 1], ['내일 뵐 수 있을지 확인 부탁드...', 0], ['도연씨. 잘 지내시죠? 오랜만 입...', 0], ['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']
df = pd.DataFrame(values, columns=columns)
X = df['메일 본문']
y = df['스팸 메일 유무']
print('X : ', X.tolist())
print('y : ', y.tolist())

# numpy 분리 
import numpy as np

np_array = np.arange(0, 16).reshape((4, 4))
print('전체 데이터 : ' , np_array)

X = np_array[:, :3]
y = np_array[:, 3] 
print(X)
print(y)