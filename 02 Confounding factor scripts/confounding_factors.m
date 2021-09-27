%% Confound investigations 
%Present the set of experiments used to adjust for confounding factrors aas
%well as investigate results 

mrisearch='^j.*.nii'; %Define the recursive search string
options.modality.name='ADNI_jac';

%% Some details to decant to a defaults function/subfunction later:
% Data
options.data.mri= '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/03_DATA/T1_PROCESSED';
options.data.cohort = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/06_ANALYSIS/000_USEFUL_SAVED-LISTS/baseline-reference-data.mat';
options.data.mask = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/04_GROUPAVERAGE/ADNI_GROUP_AVERAGE_BRAIN-MASK.nii'; %This is essential
options.output = '/scratch/gold/ttruneh 7/LD_RFS_guyon2_jac'; 
options.independent= '/home/ttruneh/ADNI_WM-6mmFWHM_SVM-INDEPENDENT-COHORT_02-Aug-2021.mat' ; %Once we have generated an independent group, we should store the path and re-use it for all the analyses - The code will save new indenpendent groups

% Options
options.mask.threshold=0.1; %For binarising 
options.cv.indpendent=0.2; %leave out 20% of data for independent test set
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

%% Load and set up data correctly
load(options.data.cohort); %Cohort data
input=cellstr(spm_select('FPListRec',options.data.mri,mrisearch)); %MRI data
exclude=zeros(numel(input),1); %Exclude those defined, or with missing data

exclude(options.exclude)=1; %Bad data on QC
exclude(root.sex==0)=1; %Missing clinical data

% Simplest possible example, two groups, AD vs not
total=sum([root.final.diagnosis==1;root.final.diagnosis==4]);
subin=[input(root.final.diagnosis==1);input(root.final.diagnosis==4)];

% Pre-allocate some matrices 
groups=zeros(total,1);age=groups;sex=groups;tiv=groups;tbv=groups;

% Allocate the data
groups(1:sum(root.final.diagnosis==1))=1; %HC
groups(sum(root.final.diagnosis==1)+1:end)=2;%AD

% Extract some other measures or confounds
age=[root.baseline.age(root.final.diagnosis==1);root.baseline.age(root.final.diagnosis==4)];
sex=[root.sex(root.final.diagnosis==1);root.sex(root.final.diagnosis==4)];
tiv=[root.mri.volumes.tiv(root.final.diagnosis==1);root.mri.volumes.tiv(root.final.diagnosis==4)];
tbv=[root.mri.volumes.tbv(root.final.diagnosis==1);root.mri.volumes.tbv(root.final.diagnosis==4)];

mridata=[input(root.final.diagnosis==1);input(root.final.diagnosis==4)];
if isempty(options.output),TIW=what;options.output=TIW.path;end
clc;close all;t1=datetime('now'); 

%% MMSE scores for each subject
%NB fairly convoluted - TODO tidy up 

%MMSE data
MMSE_table = readtable('/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/ADNI_DATA_TABLES/MMSE.csv') ;
%Select ADNI2 phase
MMSE_ADNI2 = MMSE_table(find(ismember(MMSE_table.Phase, 'ADNI2')),:) ; 

%The last 4 digits of the ADNI Subject ID (root.sid)
% correspond to the RID on the MMSE sheet 
RID = extractAfter(string(root.sid), 6) ; %extracts final 4 chars of studyID ; 

%For returning subjects --> multiple MMSE scores 
%for loop will take most recent visit MMSE score 
%TODO ideally find the date of the scan and match MMSE ....

MMSE_all = zeros(size(RID)); 

for i= 1:length(RID)
    %idx is consistent with the original subject order **
    idx = find(ismember(MMSE_ADNI2.RID, RID(i,:))) ;
    for j =1:length(idx)
        try
            MMSE_all(i,:) = str2num(MMSE_ADNI2.MMSCORE{idx(j),:});
            %skip missing entries
        end
    end
end

%**to check the correct MMSE for each subject 
check_idx = find(ismember(MMSE_ADNI2.RID, '4150')); 

%now select patients 
MMSE = [MMSE_all(root.final.diagnosis==1) ; MMSE_all(root.final.diagnosis==4) ] ;


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

%% 
X = x;Y = groups;Ui=unique(groups);G = groups; 

disp('Normalising matrix');
results.input.mu=mean(X); %Mean of voxel accross the population
results.input.stX=std(X); %Standard deviation of voxel accross the population

for i=1:size(X,1),X(i,:)=(X(i,:)-results.input.mu)./results.input.stX;end

%Currently this is not corrected for confounds, but we could put this here 

X(isnan(X))=0;

%If storing the input data
if options.writematrix
    results.input.data=X;
    results.input.groups=G;
    results.input.labels=Y;
end

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

results.input.ind=ind; %Store these allocations, important - Add in option to pass this from defaults

% Define the independent group

XI=X(ind,:);YI=Y(ind,:);
XN=X;YN=Y;GN=G;
%Remove independent data 
XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[];

%Allocate some cells
REM=cell(1,options.feature.iteration);ERROR=REM;VOXELS=ERROR;POS=ERROR;

N=nifti(M);Mi=N.dat(:,:,:);
kt=round(sum(Mi(:)>options.mask.threshold)*options.feature.removal); %How many voxels to remove

%% Adjusting for confounds - confound regression 

% See Snoek et al 2019 for conceptual overview
%Note independent data is not used at all in this process 
%Thus needs adjusting before testing

%Structures below
%N subjects, K voxels
% XN - measurement to be adjusted 
% C - design matrix size = n x (confounds +/- intercept)
% Bc - Confound parameters. size = (confounds+/- intercept ) x K 
% OLS solution Bc\hat = (C.t*C)^-1 C.t * X 

%% 01 What are the confounds? 
%Working definition: 
%'a variable that is not of primary interest, that correlates with 
%the target and is encoded in neuroimaging data' 

%  Do our 'confounds' correlate with MMSE ? 
corr_MMSE = MMSE ; corr_confound =  tbv; %insert confound HERE <

% Remove 0 vals (unrecorded) for MMSE
corr_MMSE(find(MMSE ==0)) = [] ; corr_confound(find(MMSE ==0)) = [] ; 

%Correlation coefs
[r, p] = corrcoef(corr_confound, corr_MMSE) ;

subplot(3, 2,1)
hist(corr_confound)
title('Total Brain Volume Distribution')
subplot(3,2,2)
scatter(corr_MMSE,corr_confound)
h = lsline ;
h.Color = 'r' 
xlabel('MMSE score')
ylabel('Total Brain Volume')
title('Correlation Plot')
legend('Subjects', sprintf(['r = ',num2str(r(2), 3)]));

%  Do our 'confounds' correlate with MMSE ? 
corr_MMSE = MMSE ; corr_confound =  age; %insert confound HERE <

% Remove 0 vals (unrecorded) for MMSE
corr_MMSE(find(MMSE ==0)) = [] ; corr_confound(find(MMSE ==0)) = [] ; 

%Correlation coefs
[r, p] = corrcoef(corr_confound, corr_MMSE) ;

subplot(3,2,3)
histogram((corr_confound));
title('Age Distribution')
subplot(3,2,4)
scatter(corr_MMSE,corr_confound)
h = lsline ;
h.Color = 'r' 
xlabel('MMSE score')
ylabel('Age')
title('Correlation Plot')
legend('Subjects', sprintf(['r = ',num2str(r(2), 3)]));

%  Do our 'confounds' correlate with MMSE ? 
corr_MMSE = MMSE ; corr_confound =  sex; %insert confound HERE <

% Remove 0 vals (unrecorded) for MMSE
corr_MMSE(find(MMSE ==0)) = [] ; corr_confound(find(MMSE ==0)) = [] ; 

%Correlation coefs
[r, p] = corrcoef(corr_confound, corr_MMSE) ;

subplot(3,2,5)
hist(categorical(corr_confound))
legend('Male = 1, Female = 2', 'Location', 'southeastoutside') 
title('Sex Distribution')
subplot(3,2,6)
scatter(corr_MMSE,corr_confound)
h = lsline ;
h.Color = 'r' 
xlabel('MMSE score')
ylabel('Sex')
title('Correlation Plot')
legend('Subjects', sprintf(['r = ',num2str(r(2), 3)]));


%Note - tbv is the only one that has a significant correlation 0.19 in
%isolation 

%% 02 Use confounds only as predictors 
% Uncomment whole block for basic set up

% Testing different combinations of confounds 
%confounds = [sex] ; 
confounds = [tbv]; 
%confounds = [age, sex] ; 
%confounds = [tbv, tiv] ; %NOTE tiv does not give additioal information -
%the CSF vol in this dataset is the same for all patients... 
%confounds = [tbv, tiv, age, sex] ; 

%Separate the independent confound set 
ind_confounds = confounds(ind,:) ; confounds(ind,:) = [] ; 

%Construct the design matrix 
%C = [ones(size(GN)), confounds] ; %WITH intercept 
C = [confounds] ; %WITHOUT intercept 

%Note function requires 0-1 labels (YN-1) 
%Select model to fit
%mdl = fitglm(C, (YN-1), 'distr', 'binomial', 'link', 'logit') ; %LR
%mdl = fitcdiscr(C, (YN-1), 'DiscrimType', 'linear') ; %Linear Discriminant
mdl = fitcsvm(C, (YN-1),'Standardize',true, 'KernelScale','auto', 'BoxConstraint', 'Inf'); %Hard Margin SVM 

%Test model
%WITH INTERCEPT
%labels = round(mdl.predict([ones(size(ind)), tbv(ind,:)]));
%labels = round(mdl.predict([ones(size(ind)), age(ind,:), sex(ind,:)])); 
%labels = round(mdl.predict([ones(size(ind)), tbv(ind,:), tiv(ind,:)]));
%labels = round(mdl.predict([ones(size(ind)), tbv(ind,:), tiv(ind,:), age(ind,:), sex(ind,:)])); 

%WITHout INTERCEPT
labels = round(mdl.predict([tbv(ind,:)]));
%labels = round(mdl.predict([age(ind,:), sex(ind,:)])); 
%labels = round(mdl.predict([tbv(ind,:), tiv(ind,:)])); %without intercept
%labels = round(mdl.predict([tbv(ind,:), tiv(ind,:), age(ind,:), sex(ind,:)])); %without intercept

%Metrics
accuracy = SVMbin_METRICS(labels, (Y(ind,:)-1) ); 

%RESULTS for confounds alone with logistic regression 
%tbv alone - 60% acc, 60% ba
%age, sex - 60% acc, 58% ba 
%tbv and tiv - 74% acc, 73% ba ** 
%tbv, tiv, age, sex  - 72% acc, 72% ba 
%Note highest score for SVM using tbv and tiv 77% 

%% 01 Regress out confounds from the entire train-set
% Uncomment whole block to do this
% %OLS solution for each voxel 
% Bc = inv(C'*C)*C'*XN ;
% 
% %Adjust the data 
% X_adj = XN - C*Bc ; 

%Questions
%01 ?re-standardise if we include a constant? 
%?No - Even if we have shifted everything by some constant 
%this should be consistent across voxels...
%02 ?Should each confound get it's OWN linear model and then regress out? 
%>See Rao et al 2017 

%% OR 02 fold-wise confound regression
%Snoek et al report better empirical results with this method
%Parameters are estimated on the training set of each CV fold and used to
%remove variace associated with the confounds for both the train and test
%set of that fold 

%Uncomment whole block to do this
% X_adj = zeros(size(XN));
% 
% CV_fold = 5 ;  
% [TEST, TRAIN, LABELS] = SVM_PARTITION(GN, CV_fold); 
% 
% for i = 1:CV_fold 
%     
%     B_cv = inv(C(TRAIN{i},:)'*C(TRAIN{i},:))*C(TRAIN{i},:)'*XN(TRAIN{i},:) ;
%     
%     %Adjust the training cv data
%     X_adj(TRAIN{i},:) = XN(TRAIN{i},:) - C(TRAIN{i},:)*B_cv ;
%     %Adjust the test cv data
%     X_adj(TEST{i},:) = XN(TEST{i},:) - C(TEST{i},:)*B_cv ; 
%     
% end