
mrisearch='^j.*.nii'; %Define the recursive search string
options.modality.name='ADNI_jac';

%% Some details to decant to a defaults function/subfunction later:
% Data
options.data.mri= '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/03_DATA/T1_PROCESSED';
options.data.cohort = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/06_ANALYSIS/000_USEFUL_SAVED-LISTS/baseline-reference-data.mat';
options.data.mask = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/04_GROUPAVERAGE/ADNI_GROUP_AVERAGE_BRAIN-MASK.nii'; %This is essential
options.output = '/scratch/gold/ttruneh 7/Test_features' ;
options.independent= '/scratch/gold/ttruneh 7/Test_features/independent_set';
% Options
options.mask.threshold=0.1; %For binarising 
options.cv.indpendent=0.15; %leave out 20% of data for independent test set
options.cv.kfold=10; 
options.feature.removal = 0.001; %0.004 %Remove 4% at each iteration
options.feature.iteration = 2; %Number of times to iterate STEP 1 - MRI is quite slow (but output reasonably stable if predicatable), need to implement POGS
options.feature.loopmax = 0.35; %How high the error can get before quitting
options.feature.keeperror = true; 
options.showplots = true; %Show live error
options.writeloopmri=false; %Store output from each iteration
options.smooth.window = 5; %How many points to smooth the optimisation curves (for visualising only)
options.smooth.method = 'moving'; %How to smooth the optimisation curves (for visualising only)
options.writematrix = false; %Return full normalised matrix - This can be very large
options.writeweightmri = true;

exclusion_table = readtable('/home/ttruneh/Documents/project_data/binary_bad_warps.xlsx');
options.exclude = logical(table2array(exclusion_table(:,2)));

options.random_idx = '/scratch/gold/ttruneh 7/Test_features/_random_index_23-Sep-2021.mat';  %to match class size we selected a random sample from the bigger group 
%% Load and set up data correctly
load(options.data.cohort); %Cohort data
input=cellstr(spm_select('FPListRec',options.data.mri,mrisearch)); %MRI data
exclude=zeros(numel(input),1); %Exclude those defined, or with missing data

exclude(options.exclude)=1; %Bad data on QC
exclude(root.sex==0)=1; %Missing clinical data

% Converters to AD vs non-converters 
%Matrix to inspect the groups
diagnosis_change = [root.baseline.diagnosis , root.final.diagnosis];

%subjects who ended with AD with MCI 
converters = find(root.final.diagnosis==4 & (root.baseline.diagnosis == 2| root.baseline.diagnosis ==3)  ); 

%make a logical index  from these
list = 1:numel(root.baseline.diagnosis) ; convert_idx = logical(sum((converters == list), 1))';

%$subjects who didn't end with AD with MCI
non_converters = find(root.final.diagnosis~=4 & (root.baseline.diagnosis == 2| root.baseline.diagnosis ==3)  ); 

%to match classes in size randomly sample from the non_converter group
if isempty(options.random_idx)
    rand_idx = randperm(numel(non_converters), numel(converters)); 
    
    filename=fullfile(options.output,['_random_index_',date,'.mat']);
    save(filename,'rand_idx');
    %save this 
else
    load(options.random_idx);
end

non_converters = non_converters(rand_idx); 
non_convert_idx = logical(sum((non_converters == list), 1))';

%%
total=sum([non_convert_idx;convert_idx]);
subin=[input(non_convert_idx);input(convert_idx)]; 

% Pre-allocate some matrices 
groups=zeros(total,1);age=groups;sex=groups;tiv=groups;tbv=groups;

% Allocate the data
groups(1:sum(non_convert_idx))=1; %non_converters
groups(sum(convert_idx)+1:end)=2;%converters

% Extract some other measures or confounds
age=[root.baseline.age(non_convert_idx);root.baseline.age(convert_idx)];
sex=[root.sex(non_convert_idx);root.sex(convert_idx)];
tiv=[root.mri.volumes.tiv(non_convert_idx);root.mri.volumes.tiv(convert_idx)];
tbv=[root.mri.volumes.tbv(non_convert_idx);root.mri.volumes.tbv(convert_idx)];

mridata=[input(non_convert_idx);input(convert_idx)];
if isempty(options.output),TIW=what;options.output=TIW.path;end
clc;close;t1=datetime('now'); 

%% 
% example for testing features pipeline 

%IMPORTANT 
%To use this script one must load the feature maps from the AD vs Control scripts 
%I load the whole results array and the feature maps are found 
%at results.final.reatue.voxelmap

%Can use any form of feature map but need to adjust code 

%% Read in MRI data -------------------------------------------------------
% Right then, so to start with we are going to create a 2D matrix from the
% MRI data. We are going to use a mask to only include brain voxels

M=nifti(options.data.mask);Mi=M.dat(:,:,:)>options.mask.threshold;
x = zeros(numel(mridata),sum(Mi(:))); %Pre-allocate
UD=char(strcat('Starting qmap-svmrfs',32,datestr(now)));disp(UD);
disp('Reading in MRI data'); 

for i=1:numel(mridata) 
    if ~rem(i,20),fprintf('.');end
    N=nifti(mridata{i});Ni=N.dat(:,:,:);
    x(i,:)=Ni(Mi==1);
end
disp('Done!');

%% Pre-processing

%for now just use 2 groups 
X = x;Y = groups;Ui=unique(groups);G = groups;

disp('Normalising matrix');
results_conversion.input.mu=mean(X); %Mean of voxel accross the population
results_conversion.input.stX=std(X); %Standard deviation of voxel accross the population

for i=1:size(X,1),X(i,:)=(X(i,:)-results_conversion.input.mu)./results_conversion.input.stX;end

X(isnan(X))=0;

%independent test set generation 

if isempty(options.independent) %We need to randomly allocate a balanced independent test cohort
    ind=[];N=[]; %Empty matrices in development, lots of variables in workspace!!
    
    for i=1:numel(Ui)
        N(i)=round((sum(Y==Ui(i)))*options.cv.indpendent);
        tmp=randperm(sum(Y==Ui(i)));
        count=1:1:sum(Y>0);
        tmpC=count(Y==i)';
        F=tmp(1:N(i));
        ind=[ind;tmpC(F)];
    end
    
    filename=fullfile(options.output,[options.modality.name,'_SVM-INDEPENDENT-COHORT_',date,'.mat']);
    save(filename,'ind');
else
    load(options.independent)
end
results_conversion.input.ind=ind; %Store these allocations, important - Add in option to pass this from defaults

%% 

% example for testing features pipeline 

%IMPORTANT 
%To use this script one must load the feature maps from the AD vs Control scripts 
%I load the whole results array and the feature maps are found 
%at results.final.reatue.voxelmap

%Can use any form of feature map but need to adjust code 

%% Read in MRI data -------------------------------------------------------
% Right then, so to start with we are going to create a 2D matrix from the
% MRI data. We are going to use a mask to only include brain voxels

M=nifti(options.data.mask);Mi=M.dat(:,:,:)>options.mask.threshold;
x = zeros(numel(mridata),sum(Mi(:))); %Pre-allocate
UD=char(strcat('Starting qmap-svmrfs',32,datestr(now)));disp(UD);
disp('Reading in MRI data');

for i=1:numel(mridata) 
    if ~rem(i,20),fprintf('.');end
    N=nifti(mridata{i});Ni=N.dat(:,:,:);
    x(i,:)=Ni(Mi==1);
end
disp('Done!');

%% Pre-processing

%for now just use 2 groups 
X = x;Y = groups;Ui=unique(groups);G = groups;

disp('Normalising matrix');
results_conversion.input.mu=mean(X); %Mean of voxel accross the population
results_conversion.input.stX=std(X); %Standard deviation of voxel accross the population

for i=1:size(X,1),X(i,:)=(X(i,:)-results_conversion.input.mu)./results_conversion.input.stX;end

X(isnan(X))=0;

%independent test set generation 

if isempty(options.independent) %We need to randomly allocate a balanced independent test cohort
    ind=[];N=[]; %Empty matrices in development, lots of variables in workspace!!
    
    for i=1:numel(Ui)
        N(i)=round((sum(Y==Ui(i)))*options.cv.indpendent);
        tmp=randperm(sum(Y==Ui(i)));
        count=1:1:sum(Y>0);
        tmpC=count(Y==i)';
        F=tmp(1:N(i));
        ind=[ind;tmpC(F)];
    end
    
    filename=fullfile(options.output,[options.modality.name,'_SVM-INDEPENDENT-COHORT_',date,'.mat']);
    save(filename,'ind');
else
    load(options.independent)
end

results_conversion.input.ind=ind; %Store these allocations, important - Add in option to pass this from defaults

%% 
%Create the independent test set 
XI=X(ind,:);YI=Y(ind,:);

XN=X;YN=Y;GN=G;

%remove the independent data 
XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[];

%Baseline model performance using all features 
%Linear kernel
XK=XN*XN';

disp(char(strcat('Testing standard features - Cross validation round')));
[TEST,TRAIN,LABELS]=SVM_PARTITION(GN,options.cv.kfold);

for i=1:options.cv.kfold
    
    %Transform training data - using 20 PC 
    [coeff, x_PC, latent] = pca(XK(TRAIN{i},TRAIN{i}), 'NumComponents', 20); 
    
    %fit the model on the transformed training data 
    Mdl = fitcdiscr(x_PC, YN(TRAIN{i}), 'DiscrimType', 'linear') ;

    %Use that same transform to map the test data 
    x_test_PC = XK(TEST{i},TRAIN{i})*coeff ; 

    [labels,scores] = predict(Mdl,x_test_PC);
    
    [~,~,~,AUCld] = perfcurve(YN(TEST{i}), scores(:,2),2);
    CURRENT_ERROR(i)=AUCld;
    accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
    
end

%Use all the data for the "Final Model". Store the cross validation data
%for curves/error. Put this in subfunction later
[coeff, x_PC, latent] = pca(XK,'NumComponents', 20); 
%fit the model on the transformed training data 

results_conversion.final.global.Mdl = fitcdiscr(x_PC, YN, 'DiscrimType', 'linear') ;

results_conversion.final.crossval.global.auc=CURRENT_ERROR;
results_conversion.final.crossval.global.ba=[(accuracy(:).ba)];
results_conversion.final.crossval.global.sens=[(accuracy(:).sens)];
results_conversion.final.crossval.global.spec=[(accuracy(:).spec)];
results_conversion.final.crossval.global.tp=[(accuracy(:).tp)];
results_conversion.final.crossval.global.tn=[(accuracy(:).tn)];
results_conversion.final.crossval.global.fp=[(accuracy(:).fp)];
results_conversion.final.crossval.global.fn=[(accuracy(:).fn)];

% Independent
disp(char(strcat('Testing standard features - Independent data')));
XIK=XN*(XI(:,:))';

x_test_PC = XIK'*coeff ; 
[labels,scores] = predict(results_conversion.final.global.Mdl,x_test_PC);

accuracy=SVMbin_METRICS(labels,YI);
[XSVMF,YSVMF,~,AUCsvmF] = perfcurve(YI, scores(:,2),2);
results_conversion.final.independent.global.auc=AUCsvmF;
results_conversion.final.independent.global.aucX=XSVMF;
results_conversion.final.independent.global.aucY=YSVMF;
results_conversion.final.independent.global.ba=[(accuracy(:).ba)];
results_conversion.final.independent.global.sens=[(accuracy(:).sens)];
results_conversion.final.independent.global.spec=[(accuracy(:).spec)];
results_conversion.final.independent.global.tp=[(accuracy(:).tp)];
results_conversion.final.independent.global.tn=[(accuracy(:).tn)];
results_conversion.final.independent.global.fp=[(accuracy(:).fp)];
results_conversion.final.independent.global.fn=[(accuracy(:).fn)];

%% Feature selected data - with confounds 

%LOAD results matrix with feature selected data 

disp(char(strcat('Testing SVM-RFS features - Cross validation round')));
XN=X;YN=Y;GN=G;
XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[];

% Convert from total space coordinates (feature maps) 
% to masked brain coordinates (vals)
vals = [];
for i=1:numel(results.final.feature.voxelmap)
    vals(end+1) = find(results.final.feature.voxelmap(i)==find(Mi ==1));
end

%Index voxels of interest
XN = XN(:,vals) ;

%Linear kernel
XK=XN*XN';
XIK=XN*(XI(:,vals))';

% Don't reshuffle CV, head to head
for i=1:options.cv.kfold
    
    %Transform training data - using 20 PC 
    [coeff, x_PC, latent] = pca(XK(TRAIN{i},TRAIN{i}), 'NumComponents', 20); 
    
    %fit the model on the transformed training data 
    Mdl = fitcdiscr(x_PC, YN(TRAIN{i}), 'DiscrimType', 'linear') ;

    %Use that same transform to map the test data 
    x_test_PC = XK(TEST{i},TRAIN{i})*coeff ; 

    [labels,scores] = predict(Mdl,x_test_PC);
    
    [~,~,~,AUCld] = perfcurve(YN(TEST{i}), scores(:,2),2);
    CURRENT_ERROR(i)=AUCld;
    accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
    
end

%Use all the data for the "Final Model". Store the cross validation data
%for curves/error. Put this in subfunction later
[coeff, x_PC, latent] = pca(XK,'NumComponents', 20); 
%fit the model on the transformed training data 

results_conversion.final.global.Mdl = fitcdiscr(x_PC, YN, 'DiscrimType', 'linear') ;

results_conversion.final.crossval.feature.auc=CURRENT_ERROR;
results_conversion.final.crossval.feature.ba=[(accuracy(:).ba)];
results_conversion.final.crossval.feature.sens=[(accuracy(:).sens)];
results_conversion.final.crossval.feature.spec=[(accuracy(:).spec)];
results_conversion.final.crossval.feature.tp=[(accuracy(:).tp)];
results_conversion.final.crossval.feature.tn=[(accuracy(:).tn)];
results_conversion.final.crossval.feature.fp=[(accuracy(:).fp)];
results_conversion.final.crossval.feature.fn=[(accuracy(:).fn)];

% Independent
disp(char(strcat('Testing standard features - Independent data')));

x_test_PC = XIK'*coeff ; 
[labels,scores] = predict(results_conversion.final.global.Mdl,x_test_PC);

accuracy=SVMbin_METRICS(labels,YI);

[XSVMF,YSVMF,~,AUCsvmF] = perfcurve(YI, scores(:,2),2);
results_conversion.final.independent.feature.auc=AUCsvmF;
results_conversion.final.independent.feature.aucX=XSVMF;
results_conversion.final.independent.feature.aucY=YSVMF;
results_conversion.final.independent.feature.ba=[(accuracy(:).ba)];
results_conversion.final.independent.feature.sens=[(accuracy(:).sens)];
results_conversion.final.independent.feature.spec=[(accuracy(:).spec)];
results_conversion.final.independent.feature.tp=[(accuracy(:).tp)];
results_conversion.final.independent.feature.tn=[(accuracy(:).tn)];
results_conversion.final.independent.feature.fp=[(accuracy(:).fp)];
results_conversion.final.independent.feature.fn=[(accuracy(:).fn)];


%% Feature selected data - withOUT confounds 
%LOAD results matrix with feature selected data

disp(char(strcat('Testing SVM-RFS features - Cross validation round')));
XN=X;YN=Y;GN=G;
XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[];

% Convert from total space coordinates (feature maps) 
% to masked brain coordinates (vals)
vals = [];
for i=1:numel(results.final.feature.voxelmap)
    vals(end+1) = find(results.final.feature.voxelmap(i)==find(Mi ==1));
end

%Index voxels of interest
XN = XN(:,vals) ; 

%Linear kernel
XK=XN*XN';
XIK=XN*(XI(:,vals))';

% Don't reshuffle CV, head to head
for i=1:options.cv.kfold
    
    %Transform training data - using 20 PC 
    [coeff, x_PC, latent] = pca(XK(TRAIN{i},TRAIN{i}), 'NumComponents', 20); 
    
    %fit the model on the transformed training data 
    Mdl = fitcdiscr(x_PC, YN(TRAIN{i}), 'DiscrimType', 'linear') ;

    %Use that same transform to map the test data 
    x_test_PC = XK(TEST{i},TRAIN{i})*coeff ; 

    [labels,scores] = predict(Mdl,x_test_PC);
    
    [~,~,~,AUCld] = perfcurve(YN(TEST{i}), scores(:,2),2);
    CURRENT_ERROR(i)=AUCld;
    accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
    
end

%Use all the data for the "Final Model". Store the cross validation data
%for curves/error. Put this in subfunction later
[coeff, x_PC, latent] = pca(XK,'NumComponents', 20); 
%fit the model on the transformed training data 

results_conversion.final.global.Mdl = fitcdiscr(x_PC, YN, 'DiscrimType', 'linear') ;

results_conversion.final.crossval.feature_no_conf.auc=CURRENT_ERROR;
results_conversion.final.crossval.feature_no_conf.ba=[(accuracy(:).ba)];
results_conversion.final.crossval.feature_no_conf.sens=[(accuracy(:).sens)];
results_conversion.final.crossval.feature_no_conf.spec=[(accuracy(:).spec)];
results_conversion.final.crossval.feature_no_conf.tp=[(accuracy(:).tp)];
results_conversion.final.crossval.feature_no_conf.tn=[(accuracy(:).tn)];
results_conversion.final.crossval.feature_no_conf.fp=[(accuracy(:).fp)];
results_conversion.final.crossval.feature_no_conf.fn=[(accuracy(:).fn)];

% Independent
disp(char(strcat('Testing standard features - Independent data')));

x_test_PC = XIK'*coeff ; 
[labels,scores] = predict(results_conversion.final.global.Mdl,x_test_PC);

accuracy=SVMbin_METRICS(labels,YI);

[XSVMF,YSVMF,~,AUCsvmF] = perfcurve(YI, scores(:,2),2);
results_conversion.final.independent.feature_no_conf.auc=AUCsvmF;
results_conversion.final.independent.feature_no_conf.aucX=XSVMF;
results_conversion.final.independent.feature_no_conf.aucY=YSVMF;
results_conversion.final.independent.feature_no_conf.ba=[(accuracy(:).ba)];
results_conversion.final.independent.feature_no_conf.sens=[(accuracy(:).sens)];
results_conversion.final.independent.feature_no_conf.spec=[(accuracy(:).spec)];
results_conversion.final.independent.feature_no_conf.tp=[(accuracy(:).tp)];
results_conversion.final.independent.feature_no_conf.tn=[(accuracy(:).tn)];
results_conversion.final.independent.feature_no_conf.fp=[(accuracy(:).fp)];
results_conversion.final.independent.feature_no_conf.fn=[(accuracy(:).fn)];

%% Store results 

disp('Storing results array');
filename=fullfile(options.output,[options.modality.name,'_SVMnocon-RFS-RESULTS_',date,'.mat']);
save(filename,'results_conversion');
UD=char(strcat('Finsished qmap-ldrfs',32,datestr(now)));disp(UD);t2=datetime('now');
af=cellstr(between(t1,t2));UD=char(strcat('Processing Time:',af{1}));disp(UD);

