import tensorflow as tf
import numpy as np

def generate_numpy_matrix(m, n):
    a = np.random.randint(10, size = m * n, dtype = np.int32) + 10 
    a = a.reshape(m, n) # how to do this by raw python ?
    return a

def matrix_multiply(matrix, scope):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", dtype = tf.float32, initializer=tf.truncated_normal_initializer(), shape = (256, 10))
        b = tf.get_variable("b", dtype = tf.float32, initializer=tf.truncated_normal_initializer(), shape = (10))
        y = tf.matmul(matrix, w) + b
        return y        

inputs = tf.placeholder(tf.float32, [None, 256])
y1 = matrix_multiply(inputs, "scope1")
y2 = matrix_multiply(inputs, "scope2")
result = tf.abs(y1 * y2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
matrix = generate_numpy_matrix(1, 256)
print (sess.run(result, {inputs:matrix}))

def generate(m,n):
    a = np.random.randint(10,size = 10*10,dtype=np.int32)
    a = a.reshape(m,n)
    return a
input  = tf.placeholder(tf.float32,[None,256])
w = tf.get_variable('w',dtype=tf.float32,initializer=tf.truncated_normal_initializerm(),shape=(4,3))
b = tf.get_variable('b',dtype=tf.float32,initializer=tf.truncated_normal_initializer(),shape=(3,3))
y= tf.matmul(w,input)+b
result = y
sess =tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(result,feed_dict={inpur:x3})
