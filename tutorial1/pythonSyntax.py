import numpy as np
import random
import tensorflow as tf
# x_list = [ 1,2,3,4,5,6]
#
# print(x_list)
# print(x_list[0])
# print(x_list[1:-3])
# print(x_list[0::2])
#
# x_dic = {}
# x_dic['a']=1
# x_dic['b']=2
# print(x_dic)
# print(x_dic['a'])
#
# x_set =set()
# x_set.add(1)
# x_set.add(2)
# print(x_set)
#
# def generate_matrix(m,n):
#     ls = []
#     for i in range(m):
#         sub_list = []
#         for j in range(n):
#             sub_list.append(random.randint(10,20))
#         ls.append(sub_list)
#     return ls
# a = generate_matrix(5,5)
# print(a)
# def matrix_multiply(m1,m2):
#     if len(m1[0])!=len(m2):
#         return None
#     result = []
#     for i in range(len(m1)):
#         sub = []
#         for j in range(len(m2[0])):
#             v=0
#             for k in range(len(m2)):
#                 v+= m1[i][k] * m2[k][j]
#             sub.append(v)
#         result.append(sub)
#     return  result
# matrix1 = generate_matrix(1,2)
# matrix2 = generate_matrix(2,3)
# a = matrix_multiply(matrix1,matrix2)
# print(a)

def generate_numpy_matrix(m,n):
    a = np.random.randint(10,size=m*n,dtype=np.int32)+10
    a= a.reshape(m,n)
    return a

matrix = generate_numpy_matrix(10,5)
#
# # print(matrix)
# print(matrix[0])
# print(matrix[:,0])
# print(matrix[0::2])

matrix1=generate_numpy_matrix(3,4)
matrix2=generate_numpy_matrix(3,4)
# print(matrix1)
# print(matrix2)
#
# result = np.matmul(matrix1,matrix2)
# print(result)
# result1= matrix1@matrix2
# print(result1)
#
# matrix3= generate_numpy_matrix(3,4)
# result3 =matrix1 * matrix3
# print(result3)

# print(np.transpose(matrix1))
# print(matrix1.T)
# print(np.sum(matrix1))
# print(np.sum(matrix1,axis=1))
# print(np.vstack([matrix1,matrix2]))
# print(np.hstack([matrix1,matrix2]))

v1 = tf.Variable(generate_numpy_matrix(3,4),dtype=tf.int32,name='v1')
print(v1)
v2 = tf.get_variable('v2',dtype=tf.int32,initializer=tf.constant(generate_numpy_matrix(3,4)))
print(v2)

with tf.variable_scope('scope1'):
    w1 = tf.get_variable('w1',shape=[])
    w2 =tf.Variable(0.0,name = 'w2')
with tf.variable_scope('scope1',reuse=True):
    w1_p = tf.get_variable('w1',shape=[])
    w2_p = tf.Variable(0.0,name='w2')
print(w1 is w1_p)
print(w2 is w2_p)

matrix1 = tf.get_variable('matrix1',dtype=tf.float32,initializer=tf.truncated_normal_initializer(),shape=(3,4))
matrix2 = tf.get_variable('matrix2',dtype=tf.float32,initializer=tf.truncated_normal_initializer(),shape=(4,3))
result  = tf.matmul(matrix1,matrix2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(result))