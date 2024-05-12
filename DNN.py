"""

Copyright (c) 2024 Alireza Abdi

"""

from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.layers import LeakyReLU, ReLU, BatchNormalization, Add, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from tensorflow.keras import Model
import numpy as np

def encoding(obj):
  max_len = 100
  a = 0
  c = 0.2
  g = 0.45
  t = 0.8
  z = 1
  lst = []
  for i in range(len(obj)):
    temp_lst = []
    for j in range(len(obj[i])):
      if obj[i][j] == "A":
        temp_lst.append(a)
      if obj[i][j] == "C":
        temp_lst.append(c)
      if obj[i][j] == "G":
        temp_lst.append(g)
      if obj[i][j] == "T":
        temp_lst.append(t)
    flag = 0
    if len(temp_lst) < max_len:
      flag = max_len - len(temp_lst)
    for i in range(flag):
      temp_lst.append(z)
    lst.append(temp_lst)

  lst = np.asarray(lst).astype('float32')
  return lst

class DNN_Model:
  def __init__(self, len_strings, max_length, n_action, _lr):
    self.n_action = n_action
    self.max_length = max_length
    self.len_strings = len_strings
    self.lr = _lr
    self.channel = 1
    self.model = self.build_model()
  def build_model(self):
    inx = _input = Input((self.len_strings, self.max_length, self.channel))

    model = Conv2D(64,(3, 3), strides = (1, 1),padding='same')(_input)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(128,(3, 3), strides = (1, 1),padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model0 = Conv2D(128,(3, 3), strides = (1, 1),padding='same')(model)
    model0 = BatchNormalization()(model0)
    model0 = LeakyReLU()(model0)
    model = Add()([model,model0])
    model = Conv2D(256,(3, 3), strides = (1, 1),padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model0 = Conv2D(256,(3, 3), strides = (1, 1),padding='same')(model)
    model0 = BatchNormalization()(model0)
    model0 = LeakyReLU()(model0)
    firstmodel = Add()([model,model0])

    model1 = Conv2D(64,(3, 3), strides = (1, 1),padding='same')(_input)
    model1 = BatchNormalization()(model1)
    model1 = LeakyReLU()(model1)
    model = Conv2D(128,(3, 3), strides = (1, 1),padding='same')(model1)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model0 = Conv2D(128,(3, 3), strides = (1, 1),padding='same')(model)
    model0 = BatchNormalization()(model0)
    model0 = LeakyReLU()(model0)
    model = Add()([model,model0])
    model = Conv2D(256,(3, 3), strides = (1, 1),padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model0 = Conv2D(256,(3, 3), strides = (1, 1),padding='same')(model)
    model0 = BatchNormalization()(model0)
    model0 = LeakyReLU()(model0)
    secondmodel = Add()([model,model0])

    F_Model = Concatenate()([firstmodel, secondmodel])
    P_model = Conv2D(2, (1,1), strides= (1,1), padding= 'same')(F_Model)
    P_model = BatchNormalization()(P_model)
    P_model = LeakyReLU()(P_model)
    P_model = Flatten()(P_model)
    P_model = Dense(256)(P_model)
    P_model = BatchNormalization()(P_model)
    P_model = LeakyReLU()(P_model)
    P_head = Dense(self.n_action, activation = 'softmax')(P_model)

    V_model = Conv2D(1, (1,1), strides = (1,1), padding= 'same')(F_Model)
    V_model = BatchNormalization()(V_model)
    V_model = LeakyReLU()(V_model)
    V_model = Flatten()(V_model)
    V_model = Dense(256)(V_model)
    V_model = LeakyReLU()(V_model)
    V_head = Dense(1, activation = 'tanh')(V_model)

    final_model = Model(inx, [P_head, V_head])
    
    losses = ['categorical_crossentropy', 'mean_squared_error']
    opt = Adam(learning_rate = self.lr)
    final_model.compile(loss = losses, optimizer = opt)
    return final_model

  def print_summary(self):
    print(self.model.summary())

  def set_weights(self,obj):
    self.model.set_weights(obj)

  def get_weights(self):
    return self.model.get_weights()

  def load_weights(self, fname):
    self.model.load_weights(fname)

  def save_weights(self,st):
    self.model.save_weights(st)

  def predict(self,state):
    state = encoding(state)
    state = np.expand_dims(state, 0)
    x = self.model.predict(state)
    return x

  def train(self, state, action_probability, leaf_value,counter1):
    #preparing data to fed the DNN.
    lst = []
    for i in range(len(state)):
      lst.append(encoding(state[i]))
    state = np.array(lst)
    lst1=[]
    for i in range(len(action_probability)):
      lst1.append(action_probability[i])
    action_probability = np.asarray(lst1)
    lst2 = []
    for i in range(len(leaf_value)):
      lst2.append(leaf_value[i])
    leaf_value = np.asarray(lst2)

    history_callback = self.model.fit(state, [action_probability, leaf_value], epochs = 32, batch_size = 32, verbose = 0)