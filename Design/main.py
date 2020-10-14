# -*- coding: utf-8 -*-

'''
You have to run this file to execute the Decision Tool. 

Furthermore all widgets of the interface are connected here with algorithm.
'''


import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
from copy import deepcopy
import time 


sys.path.append('../')

from Design.DesignDT_testing3 import Ui_MainWindow
from Design.DialogAbout import Ui_Dialog as Ui_Dialog_About
from Design.DialogManual import Ui_Dialog as Ui_Dialog_Manual
from ISI import Problem, Matheuristic, Meta_param, cluster_fusing, excel_to_pb


class Window(QtWidgets.QMainWindow):
    '''
        This class deals with the interface.
    '''
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
        '''
            Defines the action when clicking on the 'Browse'-button.
        '''
        self.p = self.get_path()
        self.ui.lineEdit_main.setText(self.p)               
        
        
    def pushButton_handler_opt(self):
        '''
            Defines the action when clicking on the 'Optimize'-button, 
            i.e. it reads the input file, saves the parameters set in the interface and starts solving the problem.
        '''
        self.read_from_excel(self.p)
        self.get_parameters()
        self.solveModel()
        
        self.close()    # if the window should stay open at the end, take this out
        
        
    def pushButton_handler_calculate(self):
        '''
            Defines the action when clicking on the 'Calculate'-button.
        '''
        self.get_meta_parameters()
        steps = int(np.log(self.tau_end / self.tau_start) / np.log(self.cooling))
        self.ui.lcdNumber_steps.display(steps)
        
        
    def open_wind(self):
        '''
            Opens a dialog window.
        '''
        Dialog = QtWidgets.QDialog()
        ui_dialog = Ui_Dialog_About()
        ui_dialog.setupUi(Dialog)
        Dialog.exec_()
        
    def open_man(self):
        '''
            Opens the manual dialog window when clicking on 'User Manual' and creates a link to it.
        '''
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
        '''
           Reads all the values from the input excel file.
        '''
        self.number_vehicles_used = self.ui.spinBox_veh_used.value()
        self.schools, self.warehouses,self.Q1_arr, self.V_number_input, self.makes_input = excel_to_pb(self.p, nbr_tours=self.number_vehicles_used)
    
         
    def get_path(self):
        '''
        Opens a file dialog and returns the path of the selected file as a string.

        '''
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        return path


    def get_meta_parameters(self):
        '''
            Saves the meta parameters in the third tab 'More'.
        '''
        self.tau_start = self.ui.doubleSpinBox_starttau.value()
        self.tau_end = self.ui.doubleSpinBox_endtau.value()
        self.cooling = self.ui.doubleSpinBox_cooling.value()


    def get_parameters(self):
        '''
            Saves all other parameters and deals with the checkboxes in the second tab (4 cases).
        '''
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
            if self.ui.checkBox_central.isChecked():            # Case 1: User wants to use the vehicle fleet AND the central WH of the input file.
                self.V_number = deepcopy(self.V_number_input)
                self.makes = deepcopy(self.makes_input)
                self.central = None
                self.Q1 = self.Q1_arr
                
            else:                                               # Case 2: User wants to use the vehicle fleet but NOT the central WH of the input file.
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
            if self.ui.checkBox_central.isChecked():            # Case 3: User DON'T want to use the vehicle fleet BUT want to use the central WH of the input file.
                self.central = None
                self.K = self.ui.spinBox_vehicles_wh.value()*self.number_vehicles_used
                self.Q1 = self.ui.spinBox_Q1.value()
                self.V_number = None
                self.makes = None
                
            else:                                               # Case 4: User DON'T want to use the vehicle fleet AND the central WH of the input file.
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
        '''
            Defines our problem with the parameters of the interface and solves it. 
        
            Furthermore it saves the output file.
        ------
            Raises a ValueError if no input file was selected.

        '''
        if self.ui.lineEdit_main.text()=='':
            raise ValueError('No file inserted. Could not optimize!')
            
            
        else:
            problem_global = Problem(Schools = self.schools, Warehouses = self.warehouses,
                                T = self.time_horizon, K = self.K, Q1 = self.Q1, Q2 = self.Q2, v = self.v,
                                t_load = self.t_load, c_per_km = self.c_per_km, Tmax = self.Tmax, V_number = self.V_number,
                                central = self.central, makes = self.makes, t_virt=1)

            param = Meta_param(seed=1)
            param.tau_start = self.tau_start
            param.tau_end = self.tau_end
            param.cooling = self.cooling
            param.input_var_more = [self.ui.checkBox_vehiclefleet.isChecked(), self.ui.spinBox_veh_used.value(), self.ui.spinBox_Q1.value(), self.ui.checkBox_central.isChecked(), [self.ui.doubleSpinBox_cw1.value(), self.ui.doubleSpinBox_cw2.value()], self.ui.spinBox_vehicles_central.value(), self.number_vehicles_used].copy()
            

            output_name, visu_name = create_file_names(self.p)
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            path,input_name = os.path.split(self.p)  # find the path of the directory in which there is the input sheet
            input_name = os.path.splitext(input_name)[0] # take only the name of the sheet, without the extension
            output_name = path + '/output/Output-'+input_name+ time_stamp + '.xlsx'
            visu_name = path + '/output/Visualization-'+input_name+ time_stamp + '.html'
            
            ''' 
            Finally problem gets solved.
            '''
            problem_global.final_solver(param,time_step=self.step_duration, plot_cluster = False, filename=output_name,visu_filename=visu_name)
           
            print("Optimization finished")
            print("visualisation saved in",visu_name)
            print("output sheets saved in",output_name)
            

def create_file_names(path_total):
    '''
        Creates the file names of the output.

    Parameters
    ----------
    path_total : string
        total path of the input file

    Returns
    -------
    output_name : string
        path of the excel output
    visu_name : string
        path of the visualization output

    '''
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    path1,input_name = os.path.split(path_total) # find the path of the directory in which there is the input sheet
    input_name = os.path.splitext(input_name)[0] # take only the name of the sheet, without the extension
    directory = path1+'/output'
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_name = directory + '/output/Output-'+input_name+ time_stamp + '.xlsx'
    visu_name = directory + '/output/Visualization-'+input_name+ time_stamp + '.html'

    return output_name, visu_name



if __name__ == '__main__':

    import sys
    QtWidgets.QApplication.setStyle('Fusion')
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    
