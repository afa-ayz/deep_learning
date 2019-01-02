import random
# python list
x_list = [1,2,3,4,5,6]
x_dic = {'a':1,'b':2}
x_set = set()
x_set.add(1)
# python matrix
def generate_matrix(m,n):
    ls = []
    for i in range(m):
        sub_list=[]
        for j in range(n):
            sub_list.append(random.randint(10,20))
        ls.append(sub_list)
    return  ls

def matrix_multiply(m1,m2):
    if len(m1[0]) != len(m2):
        return None
    result = []
    for i in range(len(m1)):
        sub = []
        for j in range(len(m2[0])):
            v =0
            for k in range(len(m2)):
                v += m1[i][k] * m2[k][i]
            sub.append(v)
        result.append(sub)
    return  result

# Numpy Array
import numpy as np
def generate_numpy_matrix(m,n):
    a = np.random.randint(10,size=m*n,dtype=np.int32)+10
    a = a.reshape(m,n)
    return a
# Numpy matrix multiplication
npm3= np.matrix([[1, 2], [ 3,4]])
npm1 = generate_matrix(3,2)
npm2=generate_matrix(2,3)
result = np.matmul(npm1, npm2)
# tensorflow constant
import tensorflow as tf
a = tf.constant(generate_matrix(3,4))
b = tf.constant(generate_matrix(3,4))
sess = tf.Session()
results = sess.run(a)
print(results)
# + dot mutmul place v or h
add = a + b
multiply = a * 4 + 5
dot = a * b #np.dot
b_trans = tf.transpose(b)
matmul = tf.matmul(a, b_trans)
vstack = tf.concat([a, b], axis = 0)
hstack = tf.concat([a, b], axis = 1)
sess = tf.Session()
results = sess.run([add, multiply, dot, matmul, vstack, hstack])

# tensorflow variable

v1 = tf.Variable(generate_matrix(3,4),dtype=tf.int32,name='V1')
v2 = tf.get_variable('v2',dtype=tf.int32,initializer=tf.constant(generate_matrix(3,4)))
# print(v1)
# print(v2) # get_variable reuse. Variable always create new

matrix1 = tf.get_variable('matrix1', dtype=tf.float32,initializer=tf.truncated_normal_initializer(),shape=(3,4))
matrix2 = tf.get_variable('matrix2',dtype=tf.float32,initializer=tf.truncated_normal_initializer(),shape=(4,3))
result1 = tf.matmul(matrix1, matrix2)
sess =tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(result1))

# tensorflow placeholder

a1 = tf.placeholder(tf.float32,[4,4])
a2 = tf.get_variable('a2',dtype = tf.float32 , initializer=tf.truncated_normal_initializer(),shape=(4,3))
r1 = tf.matmul(a1,a2)
sess = tf.Session()

sess.run(tf.global_variables_initializer())


a3 = generate_matrix(4,4)
print(sess.run(r1,{a1:a3}))

# tensorflow scope

with tf.variable_scope('scope1'):
    w1 = tf.get_variable('weight1',shape=[])
    print(w1)
with tf.variable_scope('scope2'):
    w1 = tf.get_variable('weight1',shape=[])
    print(w1)

def code_reuse(name):
    with tf.variable_scope(name):
        w1 = tf.get_variable('weight1',shape=[])
        print(w1)
code_reuse('scope3')
code_reuse('scope4')
