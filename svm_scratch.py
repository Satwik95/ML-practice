"""
Script based on lecture by Sentdex
source: https://www.youtube.com/watch?v=VhHLpg7ZS4Q&index=27&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

class SVM:

    __slots__ = "w", "b"
    
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        self.data = data
        # { ||w||:[w,b] }
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
        all_data = []
        opt_dict = {}

        for y in self.data:
            for X in self.data[y]:
                for x in X:
                    all_data.append(x)

        self.max_feature_val = max(all_data)
        self.min_feature_val = min(all_data)
        all_data = None

        step_sizes = [self.max_features * 0.1,
                      self.max_features * 0.01,
                      # steps get expensive at this point
                      self.max_features * 0.001]

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_val*10
        """
        Thanks to Alex Perkins youtube comment for this explaining the problem in this logic: 
        For w=[w1, w2] and w_max = 80, you should iterate through -w_max <= w1
        <= w_max  and -w_max <= w1 <= w_max independently of each other 
        Since  if w1 = w2 as in the case of the above code you're only
        searching in the vector space that at 45 degrees to the best
        separating hyperplane; as if w = [5,5] and then you iterate
        units of w equally so that w= [4,4,], [3,3], [2,2[ etc, the
        vector w will always be at 45 degrees 
        But this probably won't find the best separating hyperplane.        
        """
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   (self.max_feature_value*b_range_multiple)
                                   ,step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in SVM
                        # SMO tries to solve this
                        for i in self.data:
                            y = i
                            for x in self.data[i]:
                                if not y(np.dot(w_t,x)+b)>=1:
                                    found_option = False
                                    #break
                            if found_option:
                                # mag of vector using a linalg
                                opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                    if w[]<0:
                        optimized = True
                        print('Optimized a step')
                    else:
                        w -= step

                norms = sorted([n for n in opt_dict])
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0]+step*2

            
    def predict(self, data):
        # sign(w.x + b)
        classification = np.sign(np.dot(np.array(features),self.w)+ self.b)
        return classification
    
data_dict = {-1:np.array([[1,7],[2,8],[3,8]]),
                1:np.array([[5,1],[6,-1],[7,3]])}
