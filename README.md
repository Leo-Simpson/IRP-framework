# Last miles project : TUM case study with World Food Program

This python code solve a revisited Inventory routing problem. A user interface is also provided. 
More information is available in the technical documentation or in the user guide. 


## Files

The Design folder contains the code that generates the interface, more precisely, the file ```main.py``` should be run to launch the interface. The interface is implemented using the python package PyQt5.

The file ```ISI.py``` is the file that contains the optimization algorithm. 

The files ```OR_tools_solve_tsp.py``` and ```OR_tools_solve_tsp.py```provides auxiliary functions that are used in ``ÌSI.py```. 

THe file ```visu.py``` uses plotly package in order to 

## Dependencies of the package

Several open source python packages are needed to use the tool : 

 - sys
 - time
 - random
 - copy
 - numpy
 - pandas
 - scipy
 - sklearn
 - geopy
 - plotly
 - pulp
 - XlsxWriter
 - ortools
 - PyQt5