"""

Copyright (c) 2024 Alireza Abdi

"""

from Node import Node
import numpy as np
from math import sqrt

class mcts:
  def __init__(self,state,policy_value_net,it):
    self.state = state
    self._root = Node(self.state,0,0,0,0,'',None)
    self.policy_value_net = policy_value_net
    self.backup = []
    self.v = 0

  def select_and_expand(self, node):
    if node.number_of_children() == 0: #expansion phase
      self.backup.append(node)
      prediction = self.policy_value_net.predict(node._strings)
      self.v = prediction[1][0]
      pos_act = node.possible_action()
      alpha_var=[]
      for i in range(len(pos_act)):
        alpha_var.append(0.03)
      diri = np.random.dirichlet(alpha_var)
      if node.parent == None:
        x_temp = 0.25
      else:
        x_temp = 0.0
      for i in range(len(pos_act)):
        if pos_act[i] in node._alphabet:
          ob_str = node.move(pos_act[i])
          new_node = Node(ob_str, ((1 -x_temp) * prediction[0][0][i]) + (x_temp * diri[i]), 0, 0, 0, pos_act[i], node)
          node.add_to_children(new_node)
      return node
    #selection phase
    n = node.number_of_children()
    cpuct = 4 #2.5
    up = 0
    for i in range(n):
      up = up + node.children[i].get_N()
    up_arr = []
    for i in range(n):
      temp_x = 0
      for j in range(n):
        if i != j:
          temp_x += node.children[i].get_N()
      up_arr.append(temp_x)
    for i in range(n):
      node.children[i].set_U(cpuct * (node.children[i].get_P() * (sqrt(up) / (1 + node.children[i].get_N()))))
    x1 = node.children[0]
    cou = 0
    for i in range(n):
      if (node.children[i].get_Q()+node.children[i].get_U()) > (x1.get_Q()+x1.get_U()):
        x1 = node.children[i]
        cou = i
    return self.select_and_expand(node.children[cou])

  def bacprobagate(self,node):
    if node.parent == None:
      return 
    node.set_N(node.get_N()+1)
    node.set_W(node.get_W()+self.v)
    node.set_Q(node.get_W()/node.get_N())
    self.bacprobagate(node.parent)

  def show(self,x):
    for i in range(x.number_of_children()):
      print(x.children[i]._values._strings)
      self.show(x.children[i])

  def select_move(self):
    number = 1
    n = self._root.number_of_children()
    up = 0
    xo = []
    for i in range(n):
      up = up + (self._root.children[i].get_N()**(1 / number))
      xo.append(self._root.children[i].get_N())
    x1 = []
    x2 = []
    pos_act = self._root.possible_action()
    j = 0
    for i in range(len(self._root._alphabet)):
      if self._root._alphabet[i] in pos_act:
        x1.append((self._root.children[j].get_N()**(1 / (number))) / (up))
        x2.append(self._root.children[j].get_N())
        j += 1
      else:
        x1.append(0)
        x2.append(0)
    temp_x2 = np.array(x2)
    x3 = []
    for i in range(len(x2)):
      x3.append(0)
    x3[np.argmax(temp_x2)] = 1
    return x3

  def run(self, iter):
    for _ in range(iter):
      ndnode = self.select_and_expand(self._root)
      self.bacprobagate(ndnode)
    p = self.select_move()
    return p