# svms-from-scratch
This repository contains code for training a kernelized SVM (with multiclass
extension) in MATLAB, and specifically does not rely on any optimization libraries
(e.g. for quadratic programming). 

The SVMs are implemented using two optimization methods:
* Sequential Minimmal Optimization (SMO).
* Log Barrier with feasible start (an interior point method).
Both optimization methods optimize the dual objective formulation of SVMS, and
so the implementation is easily kernelizable. We explore Gaussian and polynomial kernels.

We test the implementation on the Historical Credit Rating dataset available directly within the MATLAB client. The main focus is numerical optimization and hence we primarliy analyse the performance of the algorithms. 

The implementation is available in ./src, and the experiments are shown in the experiments.mlx and multiclass_experiments.mlx files. A write-up (including derivation of the SVM objective, description of the algorithms, and analysis of performance) is provided in project.pdf.

A few resources that I found informative:
* http://cs229.stanford.edu/materials/smo.pdf 
* http://www0.cs.ucl.ac.uk/staff/m.pontil/reading/sv prop.pdf
* Boyd, S., & Vandenberghe, L. (2004). Convex Optimization



