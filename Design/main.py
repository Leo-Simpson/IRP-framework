# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:08:52 2020

@author: Sabrina
"""


import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import pandas as pd
#from scipy.spatial import distance_matrix, distance
import numpy as np
from copy import deepcopy


sys.path.append('../')

from Design.DesignDT_testing2 import Ui_MainWindow
from ISI import Problem, Matheuristic


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
        self.read_from_excel(p)
        
        
    def read_from_excel(self, path):
        p = path
        df_w = pd.read_excel(io=p, sheet_name='Warehouses')         #reads the excel table Warehouses
        self.warehouses = df_w.to_dict('records')                   #and transforms it into a Panda dataframe
        for w in self.warehouses:                                   #puts longitude and latitude together in a numpy array 'location'
            location = np.array([w['Latitude'],w['Longitude']])
            del w['Latitude'], w['Longitude']
            w['location']=location
            w['name'] = w.pop('Name')
            w['capacity'] = w.pop('Capacity')
            w['lower'] = w.pop('Lower')
            w['initial'] = w.pop('Initial')
            w['fixed_cost'] = w.pop('Fixed Cost')
            
    
        df_s = pd.read_excel(io=p, sheet_name='Schools')
        self.schools = df_s.to_dict('records')
        for s in self.schools:                                      #puts longitude and latitude together in a numpy array 'location'
            location = np.array([s['Latitude'],s['Longitude']])
            del s['Latitude'], s['Longitude']
            s['location']=location
            s['name'] = s.pop('Name_ID')
            s['lower'] = s.pop('Lower')
            s['initial'] = s.pop('Initial')
            s['consumption'] = s.pop('Consumption per week in mt')
            s['storage_cost'] = s.pop('Storage Cost')
            s['capacity'] = s.pop('Capacity')
            del s['Total Sum of Beneficiaries']
            del s['Total Sum of Commodities']
            del s['Consumption per day in mt']
            
        df_v = pd.read_excel(io=p, sheet_name='VehicleFleet')
        self.vehicles = df_v.to_dict('records') # list of dictionaries of the form {'Warehouse':...,'Plate Nr':....,'Make':...,'Model':....,'Capacity in MT':....}
        

        
        i = 0
        # list with N entries, which contain the list of dictionaries {'Warehouse':...., 'Plate Nr':....., 'Capacity in MT':...} per Warehouse
        # self.vehicle_list[i] gives you the list of vehicles(dictionaries) of warehouse i
        self.vehicle_list=[[] for j in range(len(self.warehouses))]
        
        for w in self.warehouses:
            for v in self.vehicles:
                if w['name'] == v['Warehouse']: 
                    v2 = deepcopy(v)
                    self.vehicle_list[i].append(v2)
                    del self.vehicle_list[i][-1]['Make'], self.vehicle_list[i][-1]['Model']    
            i+=1 
        
        self.V_number = np.array([len(self.vehicle_list[j]) for j in range(len(self.warehouses))])
        self.K_max = max(self.V_number)
        self.Q1_arr = np.zeros((len(self.warehouses), self.K_max))
        for n in range(len(self.warehouses)):
            for k in range(self.V_number[n]):
                self.Q1_arr[n,k] = self.vehicle_list[n][k]['Capacity in MT']

        
        
    
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
        self.Q2 = self.ui.spinBox_Q2.value()
        self.v = self.ui.doubleSpinBox_avgspeed.value()
        self.t_load = self.ui.doubleSpinBox_loadingtime.value()
        self.c_per_km = self.ui.doubleSpinBox_costsperkm.value()
        self.Tmax = self.ui.doubleSpinBox_maxtime.value()
        
        if self.ui.checkBox_central.isChecked():
            self.central = self.warehouses[0]
        else: 
            self.central = np.array([self.ui.doubleSpinBox_cw1.value(), self.ui.doubleSpinBox_cw2.value()])
            
        if self.ui.checkBox_vehiclefleet.isChecked():
            self.Q1 = self.Q1_arr
            self.K = None
        else: 
            self.Q1 = self.ui.spinBox_Q1.value()
            self.K = self.ui.spinBox_vehicles.value()
            self.V_number = None
            
            
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
            # and here we set up our model
            problem = Problem(Schools = self.schools, Warehouses = self.warehouses, 
                                T = self.time_horizon, K = self.K, Q1 = self.Q1, Q2 = self.Q2, v = self.v, 
                                t_load = self.t_load, c_per_km = self.c_per_km, Tmax = self.Tmax, V_number = self.V_number,
                                central = self.central, D = None)
             
            
            heuristic = Matheuristic(problem)
            heuristic.param.tau_start = 3
            heuristic.param.tau_end = 1.
            heuristic.algo2(info = True)  



if __name__ == '__main__':

    import sys
    QtWidgets.QApplication.setStyle('Fusion')
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    


