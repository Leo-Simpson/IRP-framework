# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:08:52 2020

@author: Sabrina
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from DesignDT2 import Ui_MainWindow
from PyQt5.QtWidgets import *
import pandas as pd
from ISI import Problem, Solution
from scipy.spatial import distance_matrix
import numpy as np


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.browseButton_gps.clicked.connect(self.pushButton_handler_gps)
        self.ui.browseButton_capac.clicked.connect(self.pushButton_handler_capac)
        self.ui.browseButton_cons.clicked.connect(self.pushButton_handler_cons)
        self.ui.pushButton_opt.clicked.connect(self.pushButton_handler_opt)
        
    def pushButton_handler_gps(self):
        self.write_path_gps()
    
    
    def pushButton_handler_capac(self):
        self.write_path_capac()
    
    
    def pushButton_handler_cons(self):
        self.write_path_cons()
    
        
    def pushButton_handler_opt(self):
        self.optimize()
        self.close()        
        
        
    def write_path_gps(self):
        p = self.get_path()             
        self.ui.lineEdit_gps.setText(p)             # writes path into the lineEdit
        df_w = pd.read_excel(io=p, sheet_name='Warehouses')         #reads the excel table Warehouses
        self.warehouses = df_w.to_dict('records')                   #and transforms it into a Panda dataframe
        for w in self.warehouses:                                   #puts longitude and latitude together in a numpy array 'location'
            location = np.array([w['latitude'],w['longitude']])
            del w['latitude'], w['longitude']
            w['location']=location
    
        df_s = pd.read_excel(io=p, sheet_name='Schools')
        self.schools = df_s.to_dict('records')
        for s in self.schools:                                      #puts longitude and latitude together in a numpy array 'location'
            location = np.array([s['latitude'],s['longitude']])
            del s['latitude'], s['longitude']
            s['location']=location

    
    
    def write_path_capac(self):
        p = self.get_path()
        print(p)
        self.ui.lineEdit_capac.setText(p)
        df = pd.read_excel(p)
        print(df)
    
    
    def write_path_cons(self):
        p = self.get_path()
        print(p)
        self.ui.lineEdit_cons.setText(p)
        df = pd.read_excel(p)
        print(df)
    
          
    def get_path(self):
        #opens the file dialog and returns the path's name
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        return path

    
    def get_time_horizon(self):
        self.time_horizon = self.ui.spinBox_TimeHorizon.value()
        
        
    def optimize(self):
        self.get_time_horizon()
        self.solveModel()
        
    
    def solveModel(self):
    # connects the inputs with our ISI model
        if self.ui.lineEdit_gps.text()=='':
            print('No file inserted. Could not optimize!')
        else:
            #first computes the distance matrix 
            locations = [w['location'] for w in self.warehouses] + [s['location'] for s in self.schools]
            distance_mat = distance_matrix(locations,locations)         #should we round here? Probably not, right?
            print(distance_mat)
            names = [w['name'] for w in self.warehouses] + [s['name'] for s in self.schools]
            D = pd.DataFrame(distance_mat, columns = names, index=names)
            
            #and here we set up our model
            problem = Problem(D = D, Schools = self.schools, Warehouses = self.warehouses, T = self.time_horizon, K = 5, Q1 = 1000, Q2 = 5000, v = 40, t_load = 0.5, c_per_km = 1, Tmax = 6)
            solution = Solution(problem)
            solution.ISI(G = len(self.warehouses))
        


if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

