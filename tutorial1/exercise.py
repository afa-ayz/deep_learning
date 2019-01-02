import numpy as np
import tensorflow as tf

# y = abs((w1x+b1)(w2x+b2))
# input : x (np)(shape[None,256])
# w b random variable(tf)
# output : y shape[None,10]

def generate_matrix(m,n):
    matrix = np.random.randint(10,dtype=np.int32,size=m*n)+10
    matrix = matrix.reshape(m,n)
    return matrix

def matrix_multiply(a,scope):
    with tf.variable_scope(scope):
        w = tf.get_variable('w',dtype=tf.float32,initializer=tf.truncated_normal_initializer(),shape=(256,10))
        b = tf.get_variable('b',dtype=tf.float32,initializer=tf.truncated_normal_initializer(),shape=(10))
        y = tf.matmul(a,w)+b
        return y
inputs = tf.placeholder(tf.float32,[None,256])
y1 = matrix_multiply(inputs,'scope1')
y2 = matrix_multiply(inputs,'scope2')
result = tf.abs(y1*y2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
matrix = generate_matrix(1,256)
print(sess.run(result,{inputs:matrix}))

