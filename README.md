# Value-at-Risk and Credit Cycles: Replication codes

## General Disclaimer 
These are the replication codes for the unpublished dissertation 
  *Coluccia, D. M. (2019). "Value-at-Risk and Credit Cycles: An ABM Perspective", Coluccia, D. M. (2019).*
The program is freely available upon notification to the author (davide.coluccia@unibocconi.it) and reference to the original work :bomb:

This code is part of the aforementioned dissertation, submitted in partial fullfilment to the requirements for the *Diploma Magistrale* in Economics, by Scuola Superiore Sant'Anna, to be discussed in 12/2019. Supervisor: prof. Andrea Roventini.

A copy of the work can be requested to the author after the discussion date.

## Replication Instructions
All codes are written in ```Python 3```. The ```NumPy```, ```SciPy```, ```statsmodels```, ```time```, ```pandas```, ```matplotlib```, ```csv``` and ```sklearn``` are needed to succesfully run the programs. All files need be placed in the same directory. All libraries are contained in the latest ```Anaconda``` or ```Conda``` distributions.

###### Sample Simulation
To replicate the figures of the sample simulation, run ```make_figures.py```. Estimated runtime around 20''. 
Single realization time series are obtained by setting ```T=700``` and cutting the first 550 iterations as transient (line ```120``` of ```main.py```). This is done in lines ```141-168``` of ```main.py```. 

###### Monte Carlo Experiment
To generate the Monte Carlo time series, run ```panel.py```. Monte Carlo time series are obtained by setting ```T = 750``` and cutting the first 450 iterations. This is done respectively in lines ```120``` and ```141-168``` in ```main.py```. Estimated runtime around 1h.

###### Statistical Analysis
To perform the analysis of said series, run ```MCanalysis.py```. Tables are exported in ```.tex``` format. Only run after ```.panel.py``` has been terminated.

