import pandas as pd 

###### Series #####
sr = pd.Series([170000, 18000, 1000, 5000], index = ["피자", "치킨", "콜라", "맥주"])
print(sr.values)
print(sr.index)


###### DataFrame #####
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)

print(df)
print('데이터프레임의 인덱스 : {}'.format(df.index))
print('데이터프레임의 컬럼 : {}'.format(df.columns))
print('데이터프레임의 값 : ')
print('-'*18)
print(df.values)

data = [
    ['1000', 'Steve', 90.72],
    ['1001', 'James', 78.09],
    ['1002', 'Doyeon', 98.43],
    ['1003', 'Jane', 64.19],
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14]
]

df = pd.DataFrame(data)
print(df)
df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
print(df)

# 딕셔너리로 생성하기
data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
    }

df = pd.DataFrame(data)
print(df)

#앞 부분을 3개만 보기
print(df.head(3))
print(df.tail(3))
print(df['학번'])
print(df.index)


###### Numpy #####
import numpy as np

vec = np.array([1, 2, 3, 4, 5])
mat = np.array([[1, 2, 3], [4, 5, 6]])
print(type(vec), type(mat))
print('vec의 축의 개수:', vec.ndim)
print('vec의 크기:', vec.shape)
print('mat의 축의 개수:', mat.ndim)
print('mat의 크기:', mat.shape)

zero_mat = np.zeros((3, 2))
print(zero_mat)
one_mat = np.ones((3, 2))
same_value_mat = np.full((3, 2), 7)
print(same_value_mat)
eye_mat = np.eye(3)
print(eye_mat)
random_mat = np.random.random((3, 2))

range_vec = np.arange(10)
print(range_vec)
range_n_step_vec = np.arange(1, 10, 2)
print(range_n_step_vec)

reshape_mat = np.arange(30).reshape(5, 6)
print(reshape_mat)

mat1 = np.array([[1, 2, 3], [4, 5, 6]])
slicin_mat = mat1[0, :]
print(slicin_mat)
slicin_mat = mat1[:, 1]
print(slicin_mat)