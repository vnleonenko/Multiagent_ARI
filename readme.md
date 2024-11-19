# Multiagent simulation of epidemics

This project contains code for: 
1. Preprocess population data for further multiagent simulation
2. Sample city population to speed up simulation
3. Simulation edidemic outbreak 

## Directory descriptions:
1. ```preprocessing_pipeline```
This directory contains several jupyter notebooks to
create appropriate population format from initial data gathered from 
open source. 
You can download data by link: https://disk.yandex.ru/d/BrcX8QUBlDOEug
2. ```population_sampling``` 
This directory contains jupyter notebook for sampling city population. 
It takes r% of households and leave only people from these households. 
Decreasing size of population should significantly reduce simulation time.
3. ```simulation_influenza``` 
To launch simulation run ```python example.py```. You can change parameters inside ```example.py```.