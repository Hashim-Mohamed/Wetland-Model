# 									WIL2020_WetlandsNitrogen
#					Development of python code for nitrogen processes in wetlands to support GBR Catchment modellers 
###############################################################################################################################################################################
#                                                             Trimester 1/3900ESC:WIL2020/Griffith University
#                                                               Nitrogen Cycle in Wetlands ODE Model Solver
#                                                                    Author: Hashim Mohamed (s5052848) 
#                                                                     Supervisor: Dr Melanie Roberts
###############################################################################################################################################################################
# This is python script solves a system of ODEs labeled as equations 8, 10, 11, 12, 13 in the article "Tropical Coastal Wetlands Ameliorate Nitrogen Export During Floods"[1].
# The output is in the form of numerical data generated from given input data and the ODE model
###############################################################################################################################################################################
# [1]	M.F. Adame, M.E. Roberts, D.P. Hamilton, C.E. Ndehedehe, V. Reis, J. Lu, M. Griffiths, G. Curwen and M. Ronan,
#      “Tropical Coastal Wetlands Ameliorate Nitrogen Export During Floods,” in Frontiers in Marine Science.
#      Frontiers in Marine Science, [online document], 2019. Available: Frontiers in Marine Science Online,
#      https://www.frontiersin.org/articles/10.3389/fmars.2019.00671/full [Accessed: May 24,2007].
###############################################################################################################################################################################
#
#							****Pointers on how best to use this python script & extra details****
#	(*) This code was made on Python 3
#	(*) This code uses the python modules; scipy,pandas,numpy,matplotlib and,random. Check these are installed if any issues arise. 
#	(*) The following code models the mass of nitrogen forms(SS16,PON,DON,NO3,NH4) in a wetland at any given hour.  
#	(*) An example of the input that this python script takes to produce an output can be found in the "Input" folder
#	(*) It is advised to put new input data in the existing input csv files as the name of the feilds and name of the files must remain the same for the code to operate properly.
#	(*) When entering new input data the first day should be referred to as day 0 in the input csvs.
#	(*) There is also example output in the "Output" folder.
#	(*) When the python script is run the example output will be overwritten.
#	(*) If a plot of the data is needed and not a Monte Carlo, the code to enable and save those plots can be found between lines 217 and 233(jupyter notebook) or lines 221 and 237(.py file).
#	(*) Alternatively, if you want a graph or physical representation of the output data perform a Monte Carlo with 0 uncertainty and 1 simulation.
#	(*) Values such as Crain for NO3 and NH4 as well as the rate constant ("k" and "a") values can be found between lines 54 and 96(jupyter notebook) or 58 and 100(.py file).
#	(*) The function for the interpolator can be found on line 134(jupyter notebook) or line 138(.py file).
#	(*) The ODE system can be found between lines 140 and 166(jupyter notebook) or 144 and 170(.py file) .
#	(*) The ODE solver can be found on line 181(jupyter notebook) or 185(.py file).
#	(*) The Monte Carlo code starts on line 236(jupyter notebook) or 240(.py file).
#	(*) The Monte Carlo figures come labeled with the error range that was entered e.g +/-20% and the amount of while loops or "simulations" that were done to generate the figure.
#	(*) This code can be run via the command prompt(cmd) follow (->) below  
#	-> on keyboard(Ctrl+x) -> 
#	-> select Command Prompt from pop up menu -> 
#	-> use "cd" command to navigate to this dirctory -> 
# 	-> run the script with the command "python Wetland_Nitrogen_ODE_Solver.py". 