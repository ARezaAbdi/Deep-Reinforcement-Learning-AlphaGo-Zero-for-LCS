"""

Copyright (c) 2024 Alireza Abdi

"""

class Node:
  def __init__(self, string, p, q, u, w, ch, parent):
    self._strings=string
    self.P = p
    self.Q = 0
    self.U = 0
    self.N = 0
    self.W = 0
    self.length = 0
    self.backFlag = []
    self.selected_action = [ch]
    self._alphabet = ['A','C','G','T']
    self.children = []
    self.parent = parent
    
  def add_to_children(self,obj):
    self.children.append(obj)

  def number_of_children(self):
    return len(self.children)

  def get_Q(self):
    return self.Q

  def get_U(self):
    return self.U

  def get_N(self):
    return self.N

  def get_W(self):
    return self.W

  def set_N(self,n):
    self.N=n

  def set_Q(self,q):
    self.Q=q

  def set_W(self,w):
    self.W=w

  def set_U(self,u):
    self.U=u

  def get_P(self):
    return self.P

  def possible_action(self):
    pos_act=[]
    for i in range(len(self._alphabet)):
      counter=0
      for j in range(len(self._strings)):
        x = self._strings[j].find(self._alphabet[i])
        if x >= 0:
          x = True
        else:
          x = False
        if x == True:
          counter = counter + 1
      if counter == len(self._strings):
        pos_act.append(self._alphabet[i])
      else:
        pos_act.append("x")
    return pos_act

  def move(self, action):
    st = []
    for i in range(len(self._strings)):
      x = self._strings[i].find(action)
      st.append(str(self._strings[i][x+1:]))
    return st

  def get_str(self):
    return self._strings