"""

Copyright (c) 2024 Alireza Abdi

"""

import copy
import random
from collections import deque
import DNN
from MCTS import mcts


def counting(state, alphabet):
  i = 0
  j = 0
  while i < len(state):
    if state[i] == alphabet:
      j = j + 1
    i = i + 1
  return j

def get_ub(state, alphabet):
  st = state
  ub = 0
  for j in range(len(alphabet)):
    x = []
    for k in range(len(state)):
      x.append(counting(state[k], alphabet[j]))
    temp_number = x[0]
    for k in range(len(state)):
      if x[k] < temp_number:
        temp_number = x[k]
    ub = ub + temp_number
  return int(ub)

def end(state, alphabet):
  pos_act=[]
  for i in range(len(alphabet)):
    counter = 0
    for j in range(len(state)):
      x = state[j].find(alphabet[i])
      if x >= 0:
        x = True
      if x < 0:
        x = False
      if x == True:
        counter = counter + 1
    if counter == len(state):
      pos_act.append(alphabet[i])
  if len(pos_act) == 0:
    return False
  return True

def win_or_lose(ub_value, ub_lst, lcs_length):
  counter = []
  for i in range(len(ub_lst)):
    if int(ub_value * ub_lst[i]) <= lcs_length:
      counter.append(1)
    if int(ub_value * ub_lst[i]) > lcs_length:
      counter.append(-1)
  return counter

def possible_action(_strings):
    pos_act = []
    _alphabet = ['A','C','G','T']
    for i in range(len(_alphabet)):
      counter = 0
      for j in range(len(_strings)):
        x =_strings[j].find(_alphabet[i])
        if x >= 0:
          x = True
        else:
          x = False
        if x == True:
          counter= counter + 1
      if counter == len(_strings):
        pos_act.append(_alphabet[i])
      else:
        pos_act.append("x")
    return pos_act

def do_move(state, probs, alphabet):
  probs = copy.deepcopy(probs)
  pos_act = possible_action(state)
  for i in range(len(probs)):
    if alphabet[i] not in pos_act:
      probs[i] = 0
  x = probs[0]
  c = 0
  for i in range(len(probs)):
    if probs[i] >= x:
      x = probs[i]
      c = i
  st = []
  for i in range(len(state)):
    x = state[i].find(alphabet[c])
    st.append(state[i][x + 1:])
  return st, alphabet[c] #state and action

def data_read():
  f = open("100000_100_dataset","r")
  state = f.readlines()
  for i in range(len(state)):
    state[i] = state[i].strip()
  return copy.deepcopy(state)

if __name__ == "__main__": 
  ## init ##
  change_ub_list = [0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.9]
  len_strings = 5
  max_length = 100
  learning_rate = 1e-4 #0.00001
  batch_size = 1500
  maximum_size_data_buffer = 500000
  total_iteration = 7000
  data_buffer = deque(maxlen = maximum_size_data_buffer)
  alphabet = ['A','C','G','T']
  number_of_alphabet = len(alphabet)
  main_state = data_read()
  policy_value_net = DNN.DNN_Model(len_strings, max_length, number_of_alphabet, learning_rate)
  ## init ##

  for i in range(total_iteration):
    state = random.sample(main_state, len_strings)
    probs_holder = []
    state_holder = []
    reward_holder = []
    lcs_string = ""
    lcs_length = 0
    ub = get_ub(state, alphabet)
    while end(state, alphabet):
      mcts_object = mcts(state, policy_value_net, lcs_length)
      mcts_probs = mcts_object.run(ub)
      probs_holder.append(mcts_probs)
      state_holder.append(state)
      lcs_length = lcs_length + 1
      reward_holder.append((float(get_ub(state, alphabet) / ub)))
      state, selected_letter = do_move(state, mcts_probs, alphabet)
      lcs_string = lcs_string + selected_letter
    reward_c = sum(win_or_lose(ub, change_ub_list, lcs_length)) / len(change_ub_list)
    print("LCS string: ", lcs_string, " length of LCS: ", lcs_length)
    piu = [[reward_c]]
    for it_counter in range(len(probs_holder)):
      data_buffer.append([state_holder[it_counter], probs_holder[it_counter], piu])
    if i % 250 == 0 and i != 0:
      if len(data_buffer) < batch_size:
        mini_batch = random.sample(data_buffer, len(data_buffer))
      else:
        mini_batch = random.sample(data_buffer, batch_size)
      state_batch, actions_batch, value_batch = [], [], []
      for it_counter in range(len(mini_batch)):
        state_batch.append(mini_batch[it_counter][0])
        actions_batch.append(mini_batch[it_counter][1])
        value_batch.append(mini_batch[it_counter][2])
      policy_value_net.train(state_batch,actions_batch,value_batch,i)
      fname = str(i) + '_model.h5'
      policy_value_net.save_weights(fname)
    #%tensorboard --logdir logs/fit