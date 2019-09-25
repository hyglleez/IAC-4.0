
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt


# In[2]:


def one_hot_matrix(labels, C):
  
    C = tf.constant(C,name="C")
    one_hot_matrix = tf.one_hot(labels,depth=C,axis=-1)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


# In[3]:


imagenames = []
imgs=[]
Y_label=[]

for i in range(7):
    for image in os.listdir(r"./%d"%i): #listdir
        img=imread('%d\\'%i+image)
        imgs.append(img)
        Y_label.append(i)


# In[4]:


plt.imshow(imgs[1])
plt.draw()


# In[5]:


X_train=np.array(imgs)
Y_train=np.array(Y_label)
#Y_train=tf.one_hot(Y_train,depth=7,on_value=1,off_value=0,axis=-1)
Y_train=one_hot_matrix(Y_train,7)


# In[6]:


X_train = X_train-np.mean(X_train)
Y_train.shape


# In[7]:


def weight(shape):
    initial=tf.random_normal(shape,mean=0,stddev=0.01)
    return tf.Variable(initial)
def biases1(shape):
    initial=tf.constant(1.0,shape=shape)
    return tf.Variable(initial)
def max_pool_3x3(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")


# In[8]:


x=tf.placeholder(tf.float32,[None,227,227,3],name='x') #pic size
y=tf.placeholder(tf.float32,[None,7],name='y')          # 7 lables
keep_prob = tf.placeholder(tf.float32,name='keep_prob') 

W_conv1=weight([11,11,3,96])
b_conv1=biases1([96])
x_image=tf.reshape(x,[-1,227,227,3])                   #pic size
h_conv1=tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,4,4,1],padding='VALID')+b_conv1)
h_pool1=max_pool_3x3(h_conv1)

W_conv2=weight([5,5,96,256])
b_conv2=biases1([256])
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2)
h_pool2=max_pool_3x3(h_conv2)

W_conv3=weight([3,3,256,384])
b_conv3=biases1([384])
h_conv3=tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],padding="SAME")+b_conv3)

W_conv4=weight([3,3,384,384])
b_conv4=biases1([384])
h_conv4=tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],padding="SAME")+b_conv4)

W_conv5=weight([3,3,384,256])
b_conv5=biases1([256])
h_conv5=tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv5,strides=[1,1,1,1],padding="SAME")+b_conv5)
h_pool5=max_pool_3x3(h_conv5)

nt=tf.reshape(h_pool5,[-1,9216])
W_fc1=weight([9216,4096])
b_fc1=biases1([4096]) 
h_fc1=tf.nn.relu(tf.matmul(tf.nn.dropout(nt, keep_prob),W_fc1)+b_fc1)

W_fc2=weight([4096,4096])
b_fc2=biases1([4096])
h_fc2=tf.nn.relu(tf.matmul(tf.nn.dropout(h_fc1, keep_prob),W_fc2)+b_fc2)

W_fc3=weight([4096,7])
b_fc3=biases1([7])
y_conv=tf.matmul(h_fc2,W_fc3)+b_fc3


# In[9]:


cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv,labels=y))
train=tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(cost)
step_epoch=10 # number of recursive
batch_size=32
costs=[]

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range (step_epoch):
        avg_cost=0.
        total_batch=int(X_train.shape[0]/batch_size)
       
        for i in range(total_batch):
            
            randidx=np.random.randint(X_train.shape[0],size=batch_size)
            batch_x=X_train[randidx,:,:,:]
            batch_y=Y_train[randidx,:] 
            
            _,c=sess.run([train,cost],feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
            avg_cost+=c/total_batch
            costs.append(avg_cost)
            
        if (epoch+1) % 1 ==0:
            print("Epoch:",'%04d' % (epoch+1),'cost=%9f'%avg_cost)
            print("Accuracy:",sess.run(accuracy,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0}))
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.show()
           
    print("Accuracy:",sess.run(accuracy,feed_dict={x:X_train,y:Y_train,keep_prob:1.0}))
    
    #print ("Train Accuracy:%g" %accuracy.eval(feed_dict={x: X_train, y: Y_train}))
   # print("test accuracy %g"%accuracy.eval(feed_dict={x:,y:}))


# In[59]:




