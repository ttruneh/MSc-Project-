%% SVM-Recursive Feature Elimination
%With adjustments for confounds
mrisearch='^j.*.nii'; %Define the recursive search string
options.modality.name='ADNI_jac';

%% Some details to decant to a defaults function/subfunction later:
% Data
options.data.mri= '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/03_DATA/T1_PROCESSED';
options.data.cohort = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/06_ANALYSIS/000_USEFUL_SAVED-LISTS/baseline-reference-data.mat';
options.data.mask = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/04_GROUPAVERAGE/ADNI_GROUP_AVERAGE_BRAIN-MASK.nii'; %This is essential
options.output = '/scratch/gold/~ 6/SVM_RFSguyon2_jac';
options.independent='/home/-/ADNI_WM-6mmFWHM_SVM-INDEPENDENT-COHORT_02-Aug-2021.mat' ; ; %Once we have generated an independent group, we should store the path and re-use it for all the analyses - The code will save new indenpendent groups

% Options
options.mask.threshold=0.1; %For binarising
options.cv.indpendent=0.2; %leave out 20% of data for independent test set
options.cv.kfold=10; %leave out 20% of data for independent test set
options.feature.removal = 0.003; %Remove 0.3% at each iteration
options.feature.iteration = 5; %Number of times to iterate STEP 1 - MRI is quite slow (but output reasonably stable if predicatable), need to implement POGS
options.feature.loopmax = 0.35; %How high the error can get before quitting
options.feature.keeperror = true;
options.showplots = true; %Show live error 
options.writeloopmri=true; %Store output from each iteration
options.smooth.window = 5; %How many points to smooth the optimisation curves (for visualising only)
options.smooth.method = 'moving'; %How to smooth the optimisation curves (for visualising only)
options.writematrix = false; %Return full normalised matrix - This can be very large
options.writeweightmri = true;
exclusion_table = readtable('/home/-/Documents/project_data/binary_bad_warps.xlsx');
%This excludes bad scans 
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

%% ------- confound adjustment --

X = x;Y = groups;Ui=unique(groups);G = groups;

disp('Normalising matrix');
results.input.mu=mean(X); %Mean of voxel accross the population
results.input.stX=std(X); %Standard deviation of voxel accross the population
    
for i=1:size(X,1),X(i,:)=(X(i,:)-results.input.mu)./results.input.stX;end

%Correction for confounds here -------
%Structures below
%N subjects, K voxels
% X - measurement to be adjusted 
% C - design matrix size = n x (confounds +/- intercept)
% Bc - Confound parameters. size = (confounds+/- intercept ) x K 
% OLS solution Bc\hat = (C.t*C)^-1 C.t * X 

C = [tbv, age, sex] ; 

%Fold-wise confound adjustment - see snoek 2017 
%?need to do it without the independent data?
X_adj = zeros(size(X));

CV_fold = 5 ;  
[TEST, TRAIN, LABELS] = SVM_PARTITION(G, CV_fold); 

for i = 1:CV_fold 
    
    B_cv = inv(C(TRAIN{i},:)'*C(TRAIN{i},:))*C(TRAIN{i},:)'*X(TRAIN{i},:) ;
    
    %Adjust the training cv data 
    X_adj(TRAIN{i},:) = X(TRAIN{i},:) - C(TRAIN{i},:)*B_cv ;
    %Adjust the test cv data
    X_adj(TEST{i},:) = X(TEST{i},:) - C(TEST{i},:)*B_cv ; 
    
end

% Adjust all input data
X = X_adj ;
%---------------
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

results.input.ind=ind;

%Store these allocations, important - Add in option to pass this from defaults
disp('Re-normalising matrix');
results.input.mu=mean(X); %Mean of voxel accross the population
results.input.stX=std(X); %Standard deviation of voxel accross the population
    
for i=1:size(X,1),X(i,:)=(X(i,:)-results.input.mu)./results.input.stX;end 

%% SVM-RS START
%Create the independent test set 
XI=X(ind,:);YI=Y(ind,:);

%Allocate some cells
REM=cell(1,options.feature.iteration);ERROR=REM;VOXELS=ERROR;POS=ERROR;

N=nifti(M);Mi=N.dat(:,:,:);
kt=round(sum(Mi(:)>options.mask.threshold)*options.feature.removal); %How many voxels to remove

%% Part 1 - Recursive stratification

%Recursively remove features

for pp=1:options.feature.iteration
    
    %Reset the data variables
    XN=X;YN=Y;GN=G;
    
    %Reset loop variables
    count=0;p=1;remove=[];Et=[];MODEL=[];talE=0;
    PREVIOUS_ERROR=[];CURRENT_ERROR=[];pos=[];
    STR=cell(1,options.cv.kfold);err_rate=cell(1,2);
    
    %We use a brain mask - get the voxel positions for later
    voxellist=find(Mi>options.mask.threshold);voxelpos=1:1:(size(voxellist,1));
    
    %Remove independent data
    XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[];
    
    %Linear kernel
    XK=XN*XN';
    
    %Start while loop - Will keep going until it approaches chance 
    
    while talE <options.feature.loopmax
        
        disp(char(strcat('Current Error = ',num2str(talE))));count=count+1;
        clear Mdl;rng('shuffle'); %Probably overkill =)
        
        %Set the K-fold partitions, evenly but randomly dividing the three cohorts
        [TEST,TRAIN,~]=SVM_PARTITION(GN,options.cv.kfold);
       
        %Reset/reallocate a few matrices
        tot=size(XK,2);B=zeros(tot,options.cv.kfold);dom=B; 
        
        %Let us know
        disp(char(strcat('round-',num2str(pp),';cycle-',num2str(count))));
        
        
        for i=1:options.cv.kfold
            Mdl = fitcsvm(XK(TRAIN{i},TRAIN{i}),YN(TRAIN{i}),'Standardize',true, 'KernelScale','auto','BoxConstraint', Inf);
            [labels,~] = predict(Mdl,XK(TEST{i},TRAIN{i})); %Ie. Similarity between new data and training
            
            %Error
            CURRENT_ERROR(i) = sum(YN(TEST{i})~= labels)/size(TEST{i},1);
            
            %Weights
            B(TRAIN{i},i)=B(TRAIN{i},i)+Mdl.Beta;
        end
        
        %Average K-fold weights 
        %change* no longer taking abs 
        %we will map into feature space first
        BF=mean(B,2);
        
        %BF is m by 1, XN is m by d
        W = XN'*BF ; 
        %W is d by 1
        
        BFv = W.^2;
        
        %* Remove the worst voxels
        % Sort from low to high 
        [aa,bb]=sort(BFv,'ascend'); 
        
        %Get the top voxels 
        if numel(bb)>kt
        F=bb(1:kt);
        Fvox=voxellist(F);   
        voxellist(F)=[];
        XN(:,F)=[];
        talE=mean(CURRENT_ERROR);
        end
        if numel(bb)<kt || isempty(XN)
        Fvox=voxellist;
        talE=options.feature.loopmax;
        voxellist=[];
        XN=[];
        end
                
        % Update the errors
        Et(count)=mean(CURRENT_ERROR);
        
        % If continuing to another round, remove voxels and recalculate kernel 
        if talE <options.feature.loopmax,XK=XN*XN';end
        
        if options.showplots %If we are showing plots
            subplot(2,1,1), plot(Et);drawnow;
            subplot(2,1,2),plot(smooth(Et,25),'r');drawnow;
        end
        
        % Voxels being removed and loop position
        remove=[remove;Fvox];%%%%%%%% 
        pos=[pos;(-count.*(ones(size(Fvox,1),1)))];
        CURRENT_ERROR=[];
    end
    
    % Store information from that loop
    REM{pp}=remove;
    POS{pp}=pos;
    
    if options.feature.keeperror
        results.optimising.error{pp}=Et;
    end
    
    if options.writeloopmri
        Z=zeros(size(Mi));
        Z((remove))=pos;
        if isempty(options.output),TIW=what;options.output=TIW.path;end
        filename=fullfile(options.output,[options.modality.name,'_STRUCT_SVM_TEST-removed-bin-LOOP-',num2str(pp),'.nii']);
        N.dat.fname=filename;
        N.dat.dtype='FLOAT32';
        N.dat(:,:,:)=Z;
        create(N);
    end
end

%% Part 2 - Heirachical feature addition
Z=zeros(M.dat.dim);A=Z; %Could do this with sparse matrices, being lazy

for i=1:options.feature.iteration
    Z(REM{i}) = Z(REM{i})+POS{i};
    A(REM{i}) = A(REM{i})+1;
end

%Average weighted by frequency of observation (low = important, high = 
%unimportant)
ZF=(Z./A);

if options.writeloopmri
    if isempty(options.output),TIW=what;options.output=TIW.path;end
    filename=fullfile(options.output,[options.modality.name,'_STRUCT-SVM-TEST-removed-bin-AVERAGE.nii']);
    N.dat.fname=filename;
    N.dat.dtype='FLOAT32';
    N.dat(:,:,:)=ZF;
    create(N);
end

%Now for sequential feature addition

voxellist=find(Mi>0.1);%Fx=find(Mi>0.1); 
voxelpos=1:1:(size(voxellist,1));%order=[1:1:sum(tmp)];

tmp=round(ZF(Mi>0.1));

removeBASE=find(tmp==0 | isnan(tmp));

voxelpos(removeBASE)=[];%order(tmp==0)=[]; 
voxellist(removeBASE)=[];%Fx(tmp==0)=[]; 
%
tmp(removeBASE)=[]; 
U=unique(tmp(:));

for pp=1:size(U,1)
    
    %Reset the data variables
    XN=X;YN=Y;GN=G;
    
    %Reset loop variables
    count=0;p=1;remove=[];Et=[];MODEL=[];talE=0;
    PREVIOUS_ERROR=[];CURRENT_ERROR=[];pos=[];
    STR=cell(1,options.cv.kfold);err_rate=cell(1,2);
    
    %Remove independent data
    XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[];
    
    %Remove non-feature data 
    vals=voxelpos(tmp<=U(pp)); 
    XN=XN(:,vals);
    
    %Linear kernel
    XK=XN*XN';
    
    disp(char(strcat('round-',num2str(pp))));
    [TEST,TRAIN,LABELS]=SVM_PARTITION(GN,options.cv.kfold); 
    
    for i=1:10
        Mdl = fitcsvm(XK(TRAIN{i},TRAIN{i}),YN(TRAIN{i}),'Standardize',true, 'KernelScale','auto','BoxConstraint', Inf);
        [labels,scores] = predict(Mdl,XK(TEST{i},TRAIN{i}));
        [~,~,~,AUCsvm] = perfcurve(YN(TEST{i}), scores(:,2),2);
        CURRENT_ERROR(i)=AUCsvm;
        accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
    end
    
    results.optimising.auc(pp)=mean(CURRENT_ERROR);
    results.optimising.ba(pp)=mean([(accuracy(:).ba)]);
    results.optimising.sens(pp)=mean([(accuracy(:).sens)]);
    results.optimising.spec(pp)=mean([(accuracy(:).spec)]);
    
    if options.showplots %If we are showing plots
        if pp==1,f=figure('NumberTitle', 'off', 'Name', 'SVM-RFS: ORDERING FEATURES');set(gcf,'color','w');f.Position = [100 100 1000 800];end %New figure
        subplot(2,4,1), plot(results.optimising.auc);title('AUC');drawnow;
        subplot(2,4,2),plot(results.optimising.ba);title('BA');drawnow;
        subplot(2,4,3),plot(results.optimising.sens);title('SENS');drawnow;
        subplot(2,4,4),plot(results.optimising.spec);title('SPEC');drawnow;
        subplot(2,4,5), plot(smooth(results.optimising.auc,options.smooth.window,options.smooth.method));title('SMOOTH: AUC');drawnow;
        subplot(2,4,6), plot(smooth(results.optimising.ba,options.smooth.window,options.smooth.method));title('SMOOTH: BA');drawnow;
        subplot(2,4,7), plot(smooth(results.optimising.sens,options.smooth.window,options.smooth.method));title('SMOOTH: SENS');drawnow;
        subplot(2,4,8), plot(smooth(results.optimising.spec,options.smooth.window,options.smooth.method));title('SMOOTH: SPEC');drawnow;
    end
end

if options.showplots
    %Save this plot as 300 DPI jpg
    filename=fullfile(options.output,[options.modality.name,'_SVM-FEATURE-ORDER_',date]);
    print(filename,'-djpeg','-r300');
end

%% Part 3 - Decide on number of features
% Optimal nnumber of features decided with maximum AUC. 

Fno=find(results.optimising.auc==max(results.optimising.auc));

%If there are multiple subsets performing well, we will take smallest 
Fno = min(Fno);

results.final.feature.number=Fno;

% Run k-fold on the final selection
%Reset the data variables
XN=X;YN=Y;GN=G;

%Reset variables
CURRENT_ERROR=[];clear accuracy

%Remove independent data
XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[]; 

%Linear kernel
XK=XN*XN';

disp(char(strcat('Testing standard features - Cross validation round')));
[TEST,TRAIN,LABELS]=SVM_PARTITION(GN,options.cv.kfold);

for i=1:options.cv.kfold
    Mdl = fitcsvm(XK(TRAIN{i},TRAIN{i}),YN(TRAIN{i}),'Standardize',true, 'KernelScale','auto','BoxConstraint', Inf);
    [labels,scores] = predict(Mdl,XK(TEST{i},TRAIN{i}));
    [~,~,~,AUCsvm] = perfcurve(YN(TEST{i}), scores(:,2),2);
    CURRENT_ERROR(i)=AUCsvm;
    accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
end

%Use all the data for the "Final Model". Store the cross validation data
%for curves/error. Put this in subfunction later

results.final.global.Mdl = fitcsvm(XK,YN,'Standardize',true, 'KernelScale','auto', 'BoxConstraint', Inf);
results.final.crossval.global.auc=CURRENT_ERROR;
results.final.crossval.global.ba=[(accuracy(:).ba)];
results.final.crossval.global.sens=[(accuracy(:).sens)];
results.final.crossval.global.spec=[(accuracy(:).spec)];
results.final.crossval.global.tp=[(accuracy(:).tp)];
results.final.crossval.global.tn=[(accuracy(:).tn)];
results.final.crossval.global.fp=[(accuracy(:).fp)];
results.final.crossval.global.fn=[(accuracy(:).fn)];

% Independent
disp(char(strcat('Testing standard features - Independent data')));
XIK=XN*(XI(:,:))';
[labels,scores] = predict(results.final.global.Mdl,XIK');
accuracy=SVMbin_METRICS(labels,YI);
[XSVMF,YSVMF,~,AUCsvmF] = perfcurve(YI, scores(:,2),2);
results.final.independent.global.auc=AUCsvmF;
results.final.independent.global.aucX=XSVMF;
results.final.independent.global.aucY=YSVMF;
results.final.independent.global.ba=[(accuracy(:).ba)];
results.final.independent.global.sens=[(accuracy(:).sens)];
results.final.independent.global.spec=[(accuracy(:).spec)];
results.final.independent.global.tp=[(accuracy(:).tp)];
results.final.independent.global.tn=[(accuracy(:).tn)];
results.final.independent.global.fp=[(accuracy(:).fp)];
results.final.independent.global.fn=[(accuracy(:).fn)];

% Now feature-selected data
disp(char(strcat('Testing SVM-RFS features - Cross validation round')));
XN=X;YN=Y;GN=G;
XN(ind,:)=[];YN(ind,:)=[];GN(ind)=[];

%Remove non-feature data - Long code so will reload the map

voxellist=find(Mi>0.1);%Fx=find(Mi>0.1);
voxelpos=1:1:(size(voxellist,1));%order=[1:1:sum(tmp)];

tmp=round(ZF(Mi>0.1));
removeBASE=find(tmp==0 | isnan(tmp));
tmp(removeBASE)=[];
voxelpos(removeBASE)=[];%order(tmp==0)=[];
voxellist(removeBASE)=[];%Fx(tmp==0)=[];
vals=voxelpos(tmp<=U(Fno)); %This is correct - Keep the lowest ordered features
results.final.feature.voxelmap=voxellist(tmp<=U(Fno));

XN=XN(:,vals);
%Linear kernel
XK=XN*XN';
XIK=XN*(XI(:,vals))';
% Don't reshuffle CV, head to head
for i=1:options.cv.kfold
    Mdl = fitcsvm(XK(TRAIN{i},TRAIN{i}),YN(TRAIN{i}),'Standardize',true, 'KernelScale','auto', 'BoxConstraint', Inf);
    [labels,scores] = predict(Mdl,XK(TEST{i},TRAIN{i}));
    [~,~,~,AUCsvm] = perfcurve(YN(TEST{i}), scores(:,2),2);
    CURRENT_ERROR(i)=AUCsvm;
    accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
end

results.final.feature.Mdl = fitcsvm(XK,YN,'Standardize',true, 'KernelScale','auto', 'BoxConstraint', Inf);
results.final.crossval.feature.auc=CURRENT_ERROR;
results.final.crossval.feature.ba=[(accuracy(:).ba)];
results.final.crossval.feature.sens=[(accuracy(:).sens)];
results.final.crossval.feature.spec=[(accuracy(:).spec)];
results.final.crossval.feature.tp=[(accuracy(:).tp)];
results.final.crossval.feature.tn=[(accuracy(:).tn)];
results.final.crossval.feature.fp=[(accuracy(:).fp)];
results.final.crossval.feature.fn=[(accuracy(:).fn)];

% Now independent data
disp(char(strcat('Testing SVM-RFS features - Inedpendent round')));
[labels,scores] = predict(results.final.feature.Mdl,XIK');
accuracy=SVMbin_METRICS(labels,YI);
[XSVMF,YSVMF,~,AUCsvmF] = perfcurve(YI, scores(:,2),2);
results.final.independent.feature.auc=AUCsvmF;
results.final.independent.feature.aucX=XSVMF;
results.final.independent.feature.aucY=YSVMF;
results.final.independent.feature.ba=[(accuracy(:).ba)];
results.final.independent.feature.sens=[(accuracy(:).sens)];
results.final.independent.feature.spec=[(accuracy(:).spec)];
results.final.independent.feature.tp=[(accuracy(:).tp)];
results.final.independent.feature.tn=[(accuracy(:).tn)];
results.final.independent.feature.fp=[(accuracy(:).fp)];
results.final.independent.feature.fn=[(accuracy(:).fn)];

% Finish with an AUC plot
if options.showplots
    disp('Calculating AUC plots');
    %Save this plot as 300 DPI jpg
    f=figure('NumberTitle', 'off', 'Name', 'SVM-RFS: FINAL AUC');set(gcf,'color','w');
    MdlF = fitPosterior(results.final.global.Mdl);
    [~,score_svm] = resubPredict(MdlF);
    [Xsvm,Ysvm,~,~] = perfcurve(YN,score_svm(:,2),'2');
    plot(Xsvm,Ysvm),hold on
    plot(results.final.independent.global.aucX,results.final.independent.global.aucY,'k--'),hold on
    MdlF = fitPosterior(results.final.feature.Mdl);
    [~,score_svm] = resubPredict(MdlF);
    [Xsvm,Ysvm,~,~] = perfcurve(YN,score_svm(:,2),'2');
    plot(Xsvm,Ysvm,'r');hold on;
    plot(results.final.independent.feature.aucX,results.final.independent.feature.aucY,'r--');hold on;
    legend('Global: K-fold','Global: Independent','SVM-RFS: K-fold','SVM-RFS: Independent','Location','southeast');
    filename=fullfile(options.output,[options.modality.name,'_SVM-FINAL-AUC_',date]);
    print(filename,'-djpeg','-r300');
end

%Write outthe weight map
if options.writeweightmri
    disp('Storing Weight Map');JN=zeros(size(XN));
    
    %Project weights
    for i=1:size(XK,2),JN(i,:)=results.final.feature.Mdl.Beta(i)*XN(i,:);end
    
    %Take average (voxel-wise)
    Z=zeros(M.dat.dim);Z(results.final.feature.voxelmap)=mean(JN);
    
    if isempty(options.output),TIW=what;options.output=TIW.path;end
    filename=fullfile(options.output,[options.modality.name,'STRUCT_SVM_FEATURE_WEIGHT-MAP_conf_out.nii']);
    N.dat.fname=filename;
    N.dat.dtype='FLOAT32';
    N.dat(:,:,:)=Z;
    create(N);spm_check_registration(filename);
end

disp('Storing results array');
filename=fullfile(options.output,[options.modality.name,'_SVM-RFS-RESULTS_',date,'.mat']);
save(filename,'results');
UD=char(strcat('Finsished qmap-svmrfs',32,datestr(now)));disp(UD);t2=datetime('now');
af=cellstr(between(t1,t2));UD=char(strcat('Processing Time:',af{1}));disp(UD);
