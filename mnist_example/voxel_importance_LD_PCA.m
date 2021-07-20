function [voxel_importance] = vox_importance_LD_PCA(train_kernel, train_y, train_x, n_PCA_components)
% Computes voxel importance scores (weighted norm in PCA space) for linear discriminant
%classifier with input of PCA transformed linear kernel data. See end for
%detail

%Inputs 
%train_kernel - linear kernel of training data - n_subjects X n_subjects
%train_y - labels
%train_x - samples in original feature space 
%n_PCA_components - dimension of PC space 

%Output
% Vector of importance scores for each voxel 

%Learn Principle Components 
[coeff, x_train_PC, ~] = pca(train_kernel, "NumComponents", n_PCA_components) ; 

%Fit model 
mdl = fitcdiscr(x_train_PC, train_y, 'DiscrimType', 'linear') ;

%Weighted norm of the training data in PC space 
subject_weight = abs(x_train_PC)*abs(mdl.Coeffs(1, 2).Linear) ;

%Weight the original data by it's importance in classification
voxel_importance = subject_weight'*train_x ;

end

%Rationale for this approach --------------
%1. Fitted model weights tell us how important a given feature is in the 
%discrimination task
%2. These weights are for the components of the PC space we have defined
%3. We can find the importance of each individual for the classification 
%by projecting the training data into PC space ('x_train_PC'), and weighting that
%4. Absolutes
% we will take the absolute values for model weights as well as the scores (projection into PC space) 
%as we care about overall effect on discrimination not decision itself -
%this is equivalent to a weighted norm 
%5. Weighted sum over feature space to generate a feature map which we can
%then normalize for plotting 
% -------------------------