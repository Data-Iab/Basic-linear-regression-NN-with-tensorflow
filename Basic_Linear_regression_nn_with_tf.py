import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



np.random.seed(100)

from tensorflow.keras import  Sequential
x1 = np.linspace(0,100,100).reshape(-1,1)
x2 = np.linspace(0,100,100).reshape(-1,1)
x3 = np.linspace(0,100,100).reshape(-1,1)
data = np.append(x1,np.append(x3,x2,1),1)

y =40+ 4.5*x1 +7*x2-x3+np.random.normal(0,100,size=100).reshape(-1,1)

model = Sequential([tf.keras.layers.Dense(3,input_shape=data[0].shape),
                    tf.keras.layers.Dense(1)])

model.compile(optimizer=tf.optimizers.Adam(),loss= tf.keras.losses.mean_squared_error)
model.fit(data,y,epochs=500)
plt.scatter(x1,y,color='red',marker='.')
plt.plot(x1,model.predict(data))
plt.show()