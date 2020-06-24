# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:08:52 2020

@author: Sabrina
"""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import pandas as pd
from scipy.spatial import distance_matrix
import numpy as np


sys.path.append('../')

from Design.DesignDT_testing2 import Ui_MainWindow
from ISI import Problem, Solution


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.browseButton_main.clicked.connect(self.pushButton_handler_main)
        # self.ui.browseButton_capac.clicked.connect(self.pushButton_handler_capac)
        # self.ui.browseButton_cons.clicked.connect(self.pushButton_handler_cons)
        self.ui.pushButton_opt.clicked.connect(self.pushButton_handler_opt)
        
    def pushButton_handler_main(self):
        self.write_path_main()
    
    
    # def pushButton_handler_capac(self):
    #     self.write_path_capac()
    
    
    # def pushButton_handler_cons(self):
    #     self.write_path_cons()
    
        
    def pushButton_handler_opt(self):
        self.optimize()
        # put this in if window should be closed after clicking opt button 
        # self.close()        
        
        
    def write_path_main(self):
        p = self.get_path()             
        self.ui.lineEdit_main.setText(p)             # writes path into the lineEdit
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

    
    
    # def write_path_capac(self):
    #     p = self.get_path()
    #     print(p)
    #     self.ui.lineEdit_capac.setText(p)
    #     df = pd.read_excel(p)
    #     print(df)
    
    
    # def write_path_cons(self):
    #     p = self.get_path()
    #     print(p)
    #     self.ui.lineEdit_cons.setText(p)
    #     df = pd.read_excel(p)
    #     print(df)
    
          
    def get_path(self):
        #opens the file dialog and returns the path's name
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        return path

    
    def get_parameters(self):
        self.time_horizon = self.ui.spinBox_TimeHorizon.value()
        self.K = self.ui.spinBox_vehicles.value()
        self.Q1 = self.ui.spinBox_Q1.value()
        self.Q2 = self.ui.spinBox_Q2.value()
        self.v = self.ui.doubleSpinBox_avgspeed.value()
        self.t_load = self.ui.doubleSpinBox_loadingtime.value()
        self.c_per_km = self.ui.doubleSpinBox_costsperkm.value()
        self.Tmax = self.ui.doubleSpinBox_maxtime.value()
        self.central = np.array([self.ui.doubleSpinBox_cw1.value(), self.ui.doubleSpinBox_cw2.value()])
        # print(self.time_horizon)
        # print(self.Q1)
        # print(self.Q2)
        # print(self.v)
        # print(self.t_load)
        # print(self.c_per_km)
        # print(self.Tmax)
        # print(self.central)
        
        
    def optimize(self):
        self.get_parameters()
        self.solveModel()
        
    
    def solveModel(self):
    # connects the inputs with our ISI model
        if self.ui.lineEdit_main.text()=='':
            print('No file inserted. Could not optimize!')
        else:
            #first computes the distance matrix 
            locations = [w['location'] for w in self.warehouses] + [s['location'] for s in self.schools]
            distance_mat = distance_matrix(locations,locations)         #should we round here? Probably not, right?
            #print(distance_mat)
            names = [w['name'] for w in self.warehouses] + [s['name'] for s in self.schools]
            D = pd.DataFrame(distance_mat, columns = names, index=names)
            
                
            #and here we set up our model
            problem = Problem(D = D, Schools = self.schools, Warehouses = self.warehouses, 
                              T = self.time_horizon, K = self.K, Q1 = self.Q1, Q2 = self.Q2, v = self.v, 
                              t_load = self.t_load, c_per_km = self.c_per_km, Tmax = self.Tmax, 
                              central = self.central)
            
            if self.ui.checkBox.isChecked():
                problem.central = np.mean(np.array([w["location"] for w in self.warehouses]),axis=0)
                
            solution = Solution(problem)
            solution.ISI(G = 3)

            print(solution)
        


if __name__ == '__main__':

    import sys
    QtWidgets.QApplication.setStyle('Fusion')
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    

