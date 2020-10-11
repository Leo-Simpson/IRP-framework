# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 15:34:00 2020

@author: Christophe

template: main.py
"""


import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
from copy import deepcopy


sys.path.append('../')

from Design.DesignDT_testing3 import Ui_MainWindow
from Design.DialogAbout import Ui_Dialog as Ui_Dialog_About
from Design.DialogManual import Ui_Dialog as Ui_Dialog_Manual
from ISI import Problem, Matheuristic, Meta_param, cluster_fusing, excel_to_pb


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.browseButton_main.clicked.connect(self.pushButton_handler_main)
        self.ui.pushButton_opt.clicked.connect(self.pushButton_handler_opt)
        self.ui.actionAbout.triggered.connect(self.open_wind)
        self.ui.actionUser_Manual.triggered.connect(self.open_man)
        self.ui.pushButton_calculate.clicked.connect(self.pushButton_handler_calculate)
        
        
    def pushButton_handler_main(self):
        self.p = self.get_path()
        self.ui.lineEdit_main.setText(self.p)             # writes path into the lineEdit    
        
        
    def pushButton_handler_opt(self):
        self.read_from_excel(self.p)
        self.get_parameters()
        self.solveModel()
        # put this in if window should be closed after clicking opt button
        # self.close()
        
    def pushButton_handler_calculate(self):
        self.get_meta_parameters()
        steps = int(np.log(self.tau_end / self.tau_start) / np.log(self.cooling))
        self.ui.lcdNumber_steps.display(steps)
        
        
    def open_wind(self):
        Dialog = QtWidgets.QDialog()
        ui_dialog = Ui_Dialog_About()
        ui_dialog.setupUi(Dialog)
        Dialog.exec_()
        
    def open_man(self):
        Dialog = QtWidgets.QDialog()
        ui_dialog = Ui_Dialog_Manual()
        ui_dialog.setupUi(Dialog)
        path = r"../UserManual.pdf"
        url = bytearray(QtCore.QUrl.fromLocalFile(path).toEncoded()).decode() 
        text = "<a href={}>User Manual </a>".format(url)
        ui_dialog.UserManual.setText(text)
        ui_dialog.UserManual.setOpenExternalLinks(True)
        ui_dialog.UserManual.show()
        Dialog.exec_()
        
        
    def read_from_excel(self, path):
        self.number_vehicles_used = self.ui.spinBox_veh_used.value()
        self.schools, self.warehouses,self.Q1_arr, self.V_number_input, self.makes_input = excel_to_pb(self.p, nbr_tours=self.number_vehicles_used)
    
         
    def get_path(self):
        #opens the file dialog and returns the path's name
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        return path

    def get_meta_parameters(self):
        self.tau_start = self.ui.doubleSpinBox_starttau.value()
        self.tau_end = self.ui.doubleSpinBox_endtau.value()
        self.cooling = self.ui.doubleSpinBox_cooling.value()

    def get_parameters(self):
        self.step_duration = self.ui.horizontalSlider_timeinterval.value()
        if self.step_duration == 0: self.step_duration = 0.5
        self.time_horizon = self.ui.spinBox_TimeHorizon.value()
        self.Q2 = self.ui.spinBox_Q2.value()
        self.v = self.ui.doubleSpinBox_avgspeed.value()
        self.t_load = self.ui.doubleSpinBox_loadingtime.value()
        self.c_per_km = self.ui.doubleSpinBox_costsperkm.value()
        self.Tmax = self.ui.doubleSpinBox_maxtime.value()
        self.get_meta_parameters()
        
        
        if self.ui.checkBox_vehiclefleet.isChecked():
            self.K = None
            if self.ui.checkBox_central.isChecked():
                self.V_number = deepcopy(self.V_number_input)
                self.makes = deepcopy(self.makes_input)
                self.central = None
                self.Q1 = self.Q1_arr
                
            else: 
                self.V_number = deepcopy(self.V_number_input)
                self.makes = deepcopy(self.makes_input)
                self.central = np.array([self.ui.doubleSpinBox_cw1.value(), self.ui.doubleSpinBox_cw2.value()])
                self.K_central = self.ui.spinBox_vehicles_central.value()*self.number_vehicles_used
                self.V_number = np.concatenate(([self.K_central], self.V_number), axis = 0)
                self.K_max = max(self.V_number_input)
                
                if self.K_central <= self.K_max: 
                    self.Q1_central = np.zeros((1,self.K_max))
                    makes_central = np.array([["Doesn't exist        "]*self.K_max]*1)
                    for i in range(self.K_central): 
                        self.Q1_central[0][i] = self.Q2
                        makes_central[0][i] = "Vehicle of central"
                    self.Q1 = np.concatenate((self.Q1_central, self.Q1_arr),axis = 0)
                    self.makes = np.concatenate((makes_central, self.makes), axis = 0)
                else: 
                    diff = self.K_central - self.K_max
                    zero_mat = np.zeros((len(self.warehouses), diff))
                    self.Q1_central = np.ones((1,self.K_central),dtype=float)*self.Q2
                    self.Q1_arr_ext = np.concatenate((self.Q1_arr, zero_mat), axis = 1)
                    self.Q1 = np.concatenate((self.Q1_central, self.Q1_arr_ext), axis = 0)
                    noname_makes = np.array([["Doesn't exist        "]*diff]*len(self.warehouses))
                    makes_central = np.array([["Vehicle of central" for i in range(self.K_central)]*1])
                    makes_ext = np.concatenate((self.makes, noname_makes), axis=1)
                    self.makes = np.concatenate((makes_central, makes_ext), axis=0)
                    
                    
        else: 
            if self.ui.checkBox_central.isChecked():
                self.central = None
                self.K = self.ui.spinBox_vehicles_wh.value()*self.number_vehicles_used
                self.Q1 = self.ui.spinBox_Q1.value()
                self.V_number = None
                self.makes = None
            else: 
                self.K = None
                self.central = np.array([self.ui.doubleSpinBox_cw1.value(), self.ui.doubleSpinBox_cw2.value()])
                self.K_central = self.ui.spinBox_vehicles_central.value()*self.number_vehicles_used
                K = self.ui.spinBox_vehicles_wh.value()*self.number_vehicles_used
                Q1_value = self.ui.spinBox_Q1.value()
                self.V_number = np.array([K for i in range(len(self.warehouses))])
                self.V_number = np.concatenate(([self.K_central], self.V_number), axis = 0)
                self.Q1_arr = np.ones((len(self.warehouses),K),dtype=float)*Q1_value
                makes = np.array([["No name    "]*K]*len(self.warehouses))
                
                if self.K_central <= K: 
                    self.Q1_central = np.zeros((1,K))
                    makes_central = np.array([["Doesn't exist        "]*K]*1)
                    for i in range(self.K_central): 
                        self.Q1_central[0][i] = self.Q2
                        makes_central[0][i] = "Vehicle of central"
                    self.Q1 = np.concatenate((self.Q1_central, self.Q1_arr),axis = 0)
                    self.makes = np.concatenate((makes_central, makes), axis = 0)
                else: 
                    diff = self.K_central - K
                    zero_mat = np.zeros((len(self.warehouses), diff))
                    self.Q1_central = np.ones((1,self.K_central),dtype=float)*self.Q2
                    self.Q1_arr_ext = np.concatenate((self.Q1_arr, zero_mat), axis = 1)
                    self.Q1 = np.concatenate((self.Q1_central, self.Q1_arr_ext), axis = 0)
                    noname_makes = np.array([["Doesn't exist        "]*diff]*len(self.warehouses))
                    makes_central = np.array([["Vehicle of central" for i in range(self.K_central)]*1])
                    makes_ext = np.concatenate((makes, noname_makes), axis=1)
                    self.makes = np.concatenate((makes_central, makes_ext), axis=0)
             
    
    def solveModel(self):
        # connects the inputs with our ISI model
        if self.ui.lineEdit_main.text()=='':
            raise ValueError('No file inserted. Could not optimize!')
        else:
            # and here we set up our model
            problem_global = Problem(Schools = self.schools, Warehouses = self.warehouses,
                                T = self.time_horizon, K = self.K, Q1 = self.Q1, Q2 = self.Q2, v = self.v,
                                t_load = self.t_load, c_per_km = self.c_per_km, Tmax = self.Tmax, V_number = self.V_number,
                                central = self.central, makes = self.makes, t_virt=1)

            param = Meta_param(seed=1)
            param.tau_start = self.tau_start
            param.tau_end = self.tau_end
            param.cooling = self.cooling
            param.input_var_more = [self.ui.checkBox_vehiclefleet.isChecked(), self.ui.spinBox_veh_used.value(), self.ui.spinBox_Q1.value(), self.ui.checkBox_central.isChecked(), [self.ui.doubleSpinBox_cw1.value(), self.ui.doubleSpinBox_cw2.value()], self.ui.spinBox_vehicles_central.value(), self.number_vehicles_used].copy()

            problem_global.final_solver(param,time_step=self.step_duration)

            



if __name__ == '__main__':

    import sys
    QtWidgets.QApplication.setStyle('Fusion')
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    





'''
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
            
        self.V_number_input = np.array([len(self.vehicle_list[j]) for j in range(len(self.warehouses))])
        self.K_max = max(self.V_number_input)
        self.Q1_arr = np.zeros((len(self.warehouses), self.K_max))
        for n in range(len(self.warehouses)):
            for k in range(self.V_number_input[n]):
                self.Q1_arr[n,k] = self.vehicle_list[n][k]['Capacity in MT']
'''