# Last miles project : TUM case study with World Food Program

This python code solve a revisited Inventory routing problem. A user interface is also provided. 
More information is available in the technical documentation or in the user guide. 


## Files

The Design folder contains the code that generates the interface, more precisely, the file ```main.py``` should be run to launch the interface. The interface is implemented using the python package PyQt5.

The file ```ISI.py``` is the file that contains the optimization algorithm. 

The files ```OR_tools_solve_tsp.py``` and ```OR_tools_solve_tsp.py```provides auxiliary functions that are used in ``ÌSI.py```. 

THe file ```visu.py``` uses plotly package in order to 




## Guide to run the tool using python

If one have python on his computer, (or even better, having conda installed) one can run the tool using this GitHub. 

The first step would be to download this repository. 

Then one could launch a terminal window from the folder called Design.


An optional step here that would help would be to create a separate python environment using conda (if one have conda already installed) : 
 ```shell
 conda create envLastMile
 
 source activate envLastMile
 ```
 
 Then one should make sure he can use ```pip```, the python package installer : 
 ```shell
 sudo easy_install pip
 ```
 
 Now a second step here is to install all the packages that are needed to run the python code : 
 
 ```shell 
 pip install sklearn
 
 pip install geopy
 
 pip install plotly
  
 pip install PuPL
   
 pip install XlsxWriter
    
 pip install ortools
     
 pip install PyQt5
 
 ```
 

Then one should be able to run the tool by enterring : 
```shell
python main.py
```
 
Remark : sometimes, your python app is called python3 instead, then simply run ```python3 main.py```





## Packags used in the code 

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
