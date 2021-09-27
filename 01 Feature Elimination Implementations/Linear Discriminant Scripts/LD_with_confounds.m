%% LD-RFS

mrisearch='^j.*.nii'; %Define the recursive search string
options.modality.name='ADNI_jac';

%% Some details to decant to a defaults function/subfunction later:
% Data
options.data.mri= '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/03_DATA/T1_PROCESSED';
options.data.cohort = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/06_ANALYSIS/000_USEFUL_SAVED-LISTS/baseline-reference-data.mat';
options.data.mask = '/data/underworld/01_DATA/03_OPEN-SOURCE/01_NEURODEGENERATION/ADNI2/BASELINE/04_GROUPAVERAGE/ADNI_GROUP_AVERAGE_BRAIN-MASK.nii'; %This is essential
options.output = '/scratch/gold/~ 7/LD_RFS_guyon2_jac'; 
options.independent= '/home/~/ADNI_WM-6mmFWHM_SVM-INDEPENDENT-COHORT_02-Aug-2021.mat' ; %Once we have generated an independent group, we should store the path and re-use it for all the analyses - The code will save new indenpendent groups

% Options
options.mask.threshold=0.1; %For binarising 
options.cv.indpendent=0.2; %leave out 20% of data for independent test set
options.cv.kfold=10; 
options.feature.removal = 0.003; %remove 0.3 at each iter
options.feature.iteration = 5; %Number of times to iterate STEP 1 - MRI is quite slow (but output reasonably stable if predicatable), need to implement POGS
options.feature.loopmax = 0.35; %How high the error can get before quitting
options.feature.keeperror = true; 
options.showplots = true; %Show live error
options.writeloopmri=false; %Store output from each iteration
options.smooth.window = 5; %How many points to smooth the optimisation curves (for visualising only)
options.smooth.method = 'moving'; %How to smooth the optimisation curves (for visualising only)
options.writematrix = false; %Return full normalised matrix - This can be very large
options.writeweightmri = true;

exclusion_table = readtable('/home/~/Documents/project_data/binary_bad_warps.xlsx');
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

%% 01 Linear Discriminant analysis 
%Using the MATLAB function fitcdisc - see documentation 
% help fitcdiscr 

%01 Linear Kernel - unadjusted
XK=XN*XN'; 

%% Cross validation to optimize PCA parameters 
% Using 20 for now

CV_fold = 5 ;%; 
CV_results = zeros(CV_fold, 3) ; 

[TEST, TRAIN, LABELS] = SVM_PARTITION(GN, CV_fold); 

%Kernel matrix XK is pre-computed above
comp_result = zeros(100, 1);

for comp = 1:60
    
    %1 cross validation loop 
    for i = 1:CV_fold 

        [coeff, x_PC, latent] = pca(XK(TRAIN{i},TRAIN{i}), 'NumComponents', comp); 

        %fit the model on the training PCs 
        mdl = fitcdiscr(x_PC, YN(TRAIN{i}), 'DiscrimType', 'linear') ;

        %Use that same transform to map the test data 
        x_test_PC = XK(TEST{i},TRAIN{i})*coeff ;

        %MAKE INTO FUNCTION/get rid entirely ... 
        pred_y = mdl.predict(x_test_PC) ;
        error_logical = pred_y ~=  YN(TEST{i}) ; 

        %Sensitivity  TP / TP + FN (TP we're defining as 2, the AD group) 
        TP = sum(YI(pred_y == 2) == 2) ; 
        FN = sum(YI(error_logical) == 1)  ;
        lin_sens = TP/(TP + FN); 

        %Specificity  TN / TN + FP (TN we're defining as -1) 
        TN = sum(YI(pred_y == 1) == 1) ; 
        FP = sum(YI(error_logical) == 2)  ;
        lin_spec = TN/(FP + TN);

        % Balanced accuracy = (Sens + Spec) / 2
        lin_ba = 0.5*(lin_spec + lin_sens); 

        CV_results(i, 1) = lin_ba ;  
        CV_results(i, 2) = lin_spec ;
        CV_results(i, 3) = lin_sens ;
    end

    comp_result(comp) = mean(CV_results(:,1)) ;
    
end

figure;
plot(1:100, comp_result);
title('Balanced Accuracy vs PCA components - 5 fold CV linear disc') 

%Use the number of components achieving the best result!
PCA_comp = find(comp_result == max(comp_result)) ; 

%% Metrics for independent dataset with all features 

% %If we want to run without any feature selection 
% 
% %XK computed above 
% 
% %Kernel for the independent dataset 
% 
% XIK=(XN*XI')'; %Each row is a new sample 
% 
% %If including confounds as predictors  ----------
% %Standardise the independent confounds - using the train parameters! 
% adj_ind_confounds = (ind_confounds - mean(confounds)) ./ std(confounds) ;
% XIK = [XI, adj_ind_confounds]*[XN, adj_confounds]' ;
% % -------------------------------------------------
% 
% %PCs defined on the training set
% [coeff, x_PC, latent] = pca(XK, 'NumComponents', 20); 
% 
% %Fit model on training data
% mdl = fitcdiscr(x_PC, YN, 'DiscrimType', 'linear') ;
% 
% %Transform test data
% xi_PC = XIK*coeff ;
% 
% %Confusion matrix 
% confusionchart(mdl.predict(xi_PC) , YI); 
% 
% accuracy = SVMbin_METRICS(mdl.predict(xi_PC), (Y(ind,:)) ); 
% %lin_disc_acc = 100*(1 - sum(mdl.predict(xi_PC) ~= YI) / size(YI, 1)) ; 

% See SVMbin metrics function ! 
% % error_logical = pred_y ~= YI ; 
% % 
% % %Sensitivity  TP / TP + FN (TP we're defining as 2, the AD group) 
% % TP = sum(YI(pred_y == 2) == 2) ; 
% % FN = sum(YI(error_logical) == 1)  ;
% % lin_sens = TP/(TP + FN); 
% % 
% % %Specificity  TN / TN + FP (TN we're defining as -1) 
% % TN = sum(YI(pred_y == 1) == 1) ; 
% % FP = sum(YI(error_logical) == 2)  ;
% % lin_spec = TN/(FP + TN);
% % 
% % % Balanced accuracy = (Sens + Spec) / 2
% % lin_ba = 0.5*(lin_spec + lin_sens)

%% AUC plo~ing 
% figure;
% [AUC_X, AUC_Y, T, AUC] = perfcurve(YI, pred_y ,'2'); 
% plot(AUC_X, AUC_Y) ; 

%% Feature selection - demonstration of technique to be seen on MNIST dataset script

%Rationale for voxel salience calculations 

%1. Fitted model weights tell us how important a given feature is in the 
%discrimination task
%2. These weights are for the components of the PC space we define
%3. We can find the importance of each individual for the classification 
%by projecting the training data into PC space (scores), and weighting that
%4. Absolutes 
% we will take the absolute values for model weights as well as the scores (projection into PC space) 
%as we care about overall effect on discrimination not decision itself
%5. Weighted sum over feature space to generate a feature map which we then
%normalize for plotting purposes. 

%% Part 1 - Recursive stratification 

%Same routine as for the SVM with difference in calculation of voxel
%salience 
% Recursively remove the most predictive voxels and store the order
% they were removed. Keep going until the error appoaches chance (default
% >40%). Then iterate this process, randomising k-fold at each step, to work
% out the most salient voxels. Note the independent data has been removed
% before this step and plays no part in the feature selection. 
%
%Salience of voxels computed as described above ^^
% Once the salient voxels have been removed, the kernel matrices are
% recalculated and the process repeated. Because this is voxel wise, the
% step size needs specifying. By default, this is 0.1% of all brain voxels,
% but clearly there is a speed/spatial accuracy trade-off here. 

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
        
        %Reset/reallocate a few matrices - B is for voxel saliences 
        % ------- 
        %tot=size(XK,2); B=zeros(options.cv.kfold, size(XN,2));dom=B;
        %edit 
        tot=size(XK,2);B=zeros(tot,options.cv.kfold);dom=B;
        
        %Let us know
        disp(char(strcat('round-',num2str(pp),';cycle-',num2str(count))));
        
        %Linear Discriminant Loop 
        
        for i=1:options.cv.kfold
            %Transform training data - using 20 PC for now
            [coeff, x_PC, latent] = pca(XK(TRAIN{i},TRAIN{i}), 'NumComponents', 20); 

            %fit the model on the transformed training data 
            Mdl = fitcdiscr(x_PC, YN(TRAIN{i}), 'DiscrimType', 'linear') ;

            %Use that same transform to map the test data 
            x_test_PC = XK(TEST{i},TRAIN{i})*coeff ;

            %Get Predictions 
            labels = Mdl.predict(x_test_PC) ;
            
            %Error 
            CURRENT_ERROR(i) = sum(YN(TEST{i})~= labels)/size(TEST{i},1) ;
            
            %Voxel salience 
            %Weighted norm of the training data in PC space 
            %*change - have got rid of abs for mdl coeffs and x_PC
            subject_weight = x_PC*Mdl.Coeffs(1, 2).Linear ;
            
            B(TRAIN{i},i)=B(TRAIN{i},i)+subject_weight ;

            %1043 change
%             %Weight the original data by it's importance in classification
%             voxel_salience = subject_weight'*X(TRAIN{i},:) ; 
%             
%             %Voxel salience 
%             B(i,:)=voxel_salience ;
        end
        
        %------
        
        %Average the k-fold weights 
        %*Change, no longer taking abs 
        %we will map into feature space first 
        BF=mean(B,2);
        
        %BF is m by 1, XN is m by d
        W = XN'*BF ; 
        %W is d by 1
        
        BFv = W.^2; 

        %***change now removing the worst voxels 

        % Sort from high to low 
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
        filename=fullfile(options.output,[options.modality.name,'_STRUCT_LD_TEST-removed-bin-LOOP-',num2str(pp),'.nii']);
        N.dat.fname=filename;
        N.dat.dtype='FLOAT32'; 
        N.dat(:,:,:)=Z;
        create(N);
    end
end

%% Part 2 - Heirachical feature addition 
Z=zeros(M.dat.dim);A=Z; 

for i=1:options.feature.iteration
    Z(REM{i}) = Z(REM{i})+POS{i};
    A(REM{i}) = A(REM{i})+1;
end

%Average weighted by frequency of observation (low = important, high = 
%unimportant)

ZF=(Z./A);

if options.writeloopmri
    if isempty(options.output),TIW=what;options.output=TIW.path;end
    filename=fullfile(options.output,[options.modality.name,'_STRUCT-LD-TEST-removed-bin-AVERAGE.nii']);
    N.dat.fname=filename;
    N.dat.dtype='FLOAT32';
    N.dat(:,:,:)=ZF;
    create(N);
end

voxellist=find(Mi>0.1);%Fx=find(Mi>0.1);
voxelpos=1:1:(size(voxellist,1));%order=[1:1:sum(tmp)];
tmp=round(ZF(Mi>0.1));
removeBASE=find(tmp==0 | isnan(tmp));

tmp(removeBASE)=[]; 

voxelpos(removeBASE)=[];%order(tmp==0)=[];
voxellist(removeBASE)=[];%Fx(tmp==0)=[];

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
    vals=voxelpos(tmp<=U(pp)); %This is correct - Keep the lowest ordered features
    
    XN=XN(:,vals);
    
    %Linear kernel
    XK=XN*XN';
    
    disp(char(strcat('round-',num2str(pp)))); 
    [TEST,TRAIN,LABELS]=SVM_PARTITION(GN,options.cv.kfold); 
    
    for i=1:10
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
    filename=fullfile(options.output,[options.modality.name,'_LD-FEATURE-ORDER_',date]);
    print(filename,'-djpeg','-r300');
end

%% Part 3 - Decide on number of features 
% Max auc used as criteria for addition... 

Fno=find(results.optimising.auc==max(results.optimising.auc));
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

results.final.global.Mdl = fitcdiscr(x_PC, YN, 'DiscrimType', 'linear') ;
results.final.crossval.global.auc=CURRENT_ERROR;
results.final.crossval.global.ba=[(accuracy(:).ba)];
results.final.crossval.global.sens=[(accuracy(:).sens)];
results.final.crossval.global.spec=[(accuracy(:).spec)];
results.final.crossval.global.tp=[(accuracy(:).tp)];
results.final.crossval.global.tn=[(accuracy(:).tn)];
results.final.crossval.global.tp=[(accuracy(:).fp)];
results.final.crossval.global.tn=[(accuracy(:).fn)];

% Independent
disp(char(strcat('Testing standard features - Independent data')));
XIK=XN*(XI(:,:))';

x_test_PC = XIK'*coeff ; 
[labels,scores] = predict(results.final.global.Mdl,x_test_PC);

accuracy=SVMbin_METRICS(labels,YI);
[XldF,YldF,~,AUCldF] = perfcurve(YI, scores(:,2),2);
results.final.independent.global.auc=AUCldF;
results.final.independent.global.aucX=XldF;
results.final.independent.global.aucY=YldF;
results.final.independent.global.ba=[(accuracy(:).ba)];
results.final.independent.global.sens=[(accuracy(:).sens)];
results.final.independent.global.spec=[(accuracy(:).spec)];
results.final.independent.global.tp=[(accuracy(:).tp)];
results.final.independent.global.tn=[(accuracy(:).tn)];
results.final.independent.global.tp=[(accuracy(:).fp)];
results.final.independent.global.tn=[(accuracy(:).fn)];

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

[coeff, x_PC, latent] = pca(XK,'NumComponents', 20); 
%fit the model on the transformed training data 

results.final.feature.Mdl = fitcdiscr(x_PC, YN, 'DiscrimType', 'linear') ;
results.final.crossval.feature.auc=CURRENT_ERROR;
results.final.crossval.feature.ba=[(accuracy(:).ba)];
results.final.crossval.feature.sens=[(accuracy(:).sens)];
results.final.crossval.feature.spec=[(accuracy(:).spec)];
results.final.crossval.feature.tp=[(accuracy(:).tp)];
results.final.crossval.feature.tn=[(accuracy(:).tn)];
results.final.crossval.feature.fp=[(accuracy(:).fp)];
results.final.crossval.feature.fn=[(accuracy(:).fn)];

% Now independent data
disp(char(strcat('Testing LD-RFS features - Inedpendent round')));

x_test_PC = XIK'*coeff ; 
[labels,scores] = predict(results.final.feature.Mdl,x_test_PC);
accuracy=SVMbin_METRICS(labels,YI);
[XldF,YldF,~,AUCldF] = perfcurve(YI, scores(:,2),2);
results.final.independent.feature.auc=AUCldF;
results.final.independent.feature.aucX=XldF;
results.final.independent.feature.aucY=YldF;
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
    f=figure('NumberTitle', 'off', 'Name', 'LD-RFS: FINAL AUC');set(gcf,'color','w');
    %MdlF = fitPosterior(results.final.global.Mdl);
    [~,score_ld] = resubPredict(results.final.global.Mdl);
    [Xld,Yld,~,~] = perfcurve(YN,score_ld(:,2),'2');
    plot(Xld,Yld),hold on
    plot(results.final.independent.global.aucX,results.final.independent.global.aucY,'k--'),hold on
    %MdlF = fitPosterior(results.final.feature.Mdl);
    [~,score_ld] = resubPredict(results.final.feature.Mdl);
    [Xld,Yld,~,~] = perfcurve(YN,score_ld(:,2),'2');
    plot(Xld,Yld,'r');hold on;
    plot(results.final.independent.feature.aucX,results.final.independent.feature.aucY,'r--');hold on;
    legend('Global: K-fold','Global: Independent','LD-RFS: K-fold','LD-RFS: Independent','Location','southeast');
    filename=fullfile(options.output,[options.modality.name,'_LD-FINAL-AUC_',date]);
    print(filename,'-djpeg','-r300');
end

%Write out the weight map 
if options.writeweightmri
    disp('Storing Weight Map');JN=zeros(size(XN));
    
    %To recap
    %We trained the model on a reduced dim feature set (dim = 20) 
    %Model coefficients are with respect to each dimension (n = 20)
    %We should be able to get the subject weight through transformed data * coeff

    subject_weight = x_PC*results.final.feature.Mdl.Coeffs(1, 2).Linear ;
    
    %Project weights back into the original feature space
    for i=1:size(XK,2),JN(i,:)=subject_weight(i)*XN(i,:);end
    
    %Take average (voxel-wise)
    Z=zeros(M.dat.dim);Z(results.final.feature.voxelmap)=-1*mean(JN)';
    
    if isempty(options.output),TIW=what;options.output=TIW.path;end
    filename=fullfile(options.output,[options.modality.name,'STRUCT_LD_FEATURE_WEIGHT-MAP.nii']);
    N.dat.fname=filename; 
    N.dat.dtype='FLOAT32';
    N.dat(:,:,:)=Z;
    create(N);spm_check_registration(filename);
end

disp('Storing results array');
filename=fullfile(options.output,[options.modality.name,'_LD-RFS-RESULTS_',date,'.mat']);
save(filename,'results');
UD=char(strcat('Finsished qmap-ldrfs',32,datestr(now)));disp(UD);t2=datetime('now');
af=cellstr(between(t1,t2));UD=char(strcat('Processing Time:',af{1}));disp(UD);
