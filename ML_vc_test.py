#!/usr/bin/env python
# coding: utf-8

# In[1]:


from trains import Task


# In[2]:


task = Task.init(project_name="Project_1", task_name="sk_task1")


# In[3]:


# Trains - Example of integrating plots and training on jupyter notebook. 
# In this example, simple graphs are shown, then an MNIST classifier is trained using Keras.

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt


# In[4]:


# Set script parameters
task_params = {'num_scatter_samples': 50, 'sin_max_value': 30, 'sin_steps': 25}
task_params = task.connect(task_params)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
N = task_params['num_scatter_samples']
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (50 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Nice Circles1')
plt.show()

x = np.linspace(0, task_params['sin_max_value'], task_params['sin_steps'])
y = np.sin(x)
plt.plot(x, y, 'o', color='black')
plt.title('Sinus Dots')
plt.show()

m = np.eye(32, 32, dtype=np.uint8)
plt.imshow(m)
plt.title('sample output')
plt.show()


# In[28]:


# Notice, Updating task_params is traced and updated in TRAINS
task_params['batch_size'] = 128
task_params['nb_classes'] = 10
task_params['nb_epoch'] = 6
task_params['hidden_dim'] = 512
batch_size = task_params['batch_size']
nb_classes = task_params['nb_classes']
nb_epoch = task_params['nb_epoch']


# In[27]:


print('batch_size' , batch_size,'nb_classes' , nb_classes,'nb_epoch' , nb_epoch)


# In[29]:


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

hidden_dim = task_params['hidden_dim']
model1 = Sequential()
model1.add(Dense(hidden_dim, input_shape=(784,)))
model1.add(Activation('relu'))
# model.add(Dropout(0.2))
model1.add(Dense(hidden_dim))
model1.add(Activation('relu'))
# model.add(Dropout(0.2))
model1.add(Dense(10))
model1.add(Activation('softmax'))

model1.summary()

model1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

board = TensorBoard(histogram_freq=1, log_dir='/tmp/histogram_example')
model1_store = ModelCheckpoint(filepath='/tmp/weight.{epoch}.hdf5')

model1.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    callbacks=[board, model_store],
                    verbose=1, validation_data=(X_test, Y_test))
score = model1.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[31]:


from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics 


# In[34]:


get_ipython().system('pip3 install -U xgboost')


# In[35]:


import xgboost
model_h = xgboost.XGBRegressor(n_estimators=70, learning_rate=0.08, gamma=0, subsample=0.8,
                               colsample_bytree=1, max_depth=10)


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[ ]:





# In[10]:


X, y = make_classification(n_samples=1000, n_features=4,
                          n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                            random_state=0)
clf.fit(X, y)  
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))


# In[ ]:




