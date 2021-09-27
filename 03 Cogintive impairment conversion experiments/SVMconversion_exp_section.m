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

%% Single run of SVM to test 

%for now just use 2 groups 
X = x;Y = groups;Ui=unique(groups);

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
    Mdl = fitcsvm(XK(TRAIN{i},TRAIN{i}),YN(TRAIN{i}),'Standardize',true, 'KernelScale','auto','BoxConstraint', Inf);
    [labels,scores] = predict(Mdl,XK(TEST{i},TRAIN{i}));
    [~,~,~,AUCsvm] = perfcurve(YN(TEST{i}), scores(:,2),2);
    CURRENT_ERROR(i)=AUCsvm;
    accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
end

%Save results 
%Use all the data for the "Final Model". Store the cross validation data
%for curves/error. Put this in subfunction later

results_conversion.final.global.Mdl = fitcsvm(XK,YN,'Standardize',true, 'KernelScale','auto', 'BoxConstraint', Inf);
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
[labels,scores] = predict(results_conversion.final.global.Mdl,XIK');
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

% Now feature-selected data

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
    Mdl = fitcsvm(XK(TRAIN{i},TRAIN{i}),YN(TRAIN{i}),'Standardize',true, 'KernelScale','auto', 'BoxConstraint', Inf);
    [labels,scores] = predict(Mdl,XK(TEST{i},TRAIN{i}));
    [~,~,~,AUCsvm] = perfcurve(YN(TEST{i}), scores(:,2),2);
    CURRENT_ERROR(i)=AUCsvm;
    accuracy(i)=SVMbin_METRICS(labels,YN(TEST{i}));
end

results_conversion.final.feature.Mdl = fitcsvm(XK,YN,'Standardize',true, 'KernelScale','auto', 'BoxConstraint', Inf);
results_conversion.final.crossval.feature.auc=CURRENT_ERROR;
results_conversion.final.crossval.feature.ba=[(accuracy(:).ba)];
results_conversion.final.crossval.feature.sens=[(accuracy(:).sens)];
results_conversion.final.crossval.feature.spec=[(accuracy(:).spec)];
results_conversion.final.crossval.feature.tp=[(accuracy(:).tp)];
results_conversion.final.crossval.feature.tn=[(accuracy(:).tn)];
results_conversion.final.crossval.feature.fp=[(accuracy(:).fp)];
results_conversion.final.crossval.feature.fn=[(accuracy(:).fn)];

% Now independent data
disp(char(strcat('Testing SVM-RFS features - Inedpendent round')));
[labels,scores] = predict(results_conversion.final.feature.Mdl,XIK');
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


%%
