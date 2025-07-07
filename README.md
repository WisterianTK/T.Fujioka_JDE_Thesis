# T.Fujioka_JDE_Thesis
Master Thesis:  Joint Estimation of Neural and Hemodynamic Responses in Functional Ultrasound Using Variational Inference

To replicate the real experiment in my thesis, you need to visit the dataset from Nunez-Elizalde et. al at "https://figshare.com/projects/Nunez-Elizalde2022/132110".
The data is stored in hdf format so you need to convert this to mat format.
To do this, first download repository at "https://github.com/anwarnunez/fusi". 
Then, place "get_MATLAB_data" in this repository to "scripts" in the repository from Nunez-Elizalde.
The converted data should appear in "Subjects_for_MATLAB" in the downloaded repository from Nunez-Elizalde.
Finally, move "Subjects_for_MATLAB" directory to this repository you downloaded. 
(If the conversion fail, it is due to missing files called "protocol.mat" in the dataset. Unfortunately I cannot share this file so please contact the lead contact person mentioned in the paper "Neural correlates of blood flow measured by ultrasound" by Nunez-Elizalde et. al.)
