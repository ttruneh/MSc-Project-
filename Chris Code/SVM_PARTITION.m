function [TEST,TRAIN,LABELS]=SVM_PARTITION(GROUPS,FOLDS,MERGE)
%function [TEST,TRAIN,LABELS]=SVM_PARTITION(GROUPS,FOLDS,MERGE)
%Create balanced training groups (proportional to original cohort). If wanting
%balanced 50:50 splits, sort groups prior to this point. Former not an issue
%provided balanced accuracy used instead of accuracy (otherwise 50% chance
%does not hold true), and makes data partition much cleaner! 
% 
%Essential inputs:
%GROUPS = 1xn vector of group allocation, 1..n
%FOLDS = k fold validation, define k
%MERGE = If Groups >2, define which ones to merge to create binary problem,
%Supply as a cell, allows for multiple groups within each pool.
%
%Output:
%TEST - 1 to k cell of positions for test data
%TRAIN - 1 to k cell of posisions for training data
%LABELS - Binary case will == Groups, merged will be combined groupings
%--------------------------------------------------------------------------
%C Lambert - Version 1.0 - Sept 2017
%--------------------------------------------------------------------------

%Seed random number generator
rng('shuffle');

%PARTITION FRACTION
Ks=1/FOLDS;

%Check inputs
if nargin<3
    MERGE=[];
    if nargin<2
        disp('ERROR - INSUFFICIENT INPUTS');
        TEST=NaN;TRAIN=NaN;LABELS=NaN;
        return
    end
end

%First lets figure out groups and sizes
GROUPS=GROUPS(:);uG=unique(GROUPS);
G=zeros(size(GROUPS));uG=uG(:);sG=zeros(size(uG));
for i=1:size(uG,1),sG(i)=sum(GROUPS==uG(i));end

if isempty(MERGE) %Lets do this in two stages, first classic binary problem
    
    %Quick check
    if FOLDS>min(sG)
        disp('Error - More folds than data available');
        TEST=NaN;TRAIN=NaN;LABELS=NaN;
        return
    end
    
    for i=1:size(uG,1),
        G(GROUPS==uG(i))=i;
    end
    
    GP=1:1:size(G,1); Gs=cell(1,2);Go=Gs;
    
    %Figure out data proportions
    for j=1:2
        Gs{j}=round(sG(j)*Ks);Go{j}=randperm(sG(j));
    end
    
    for i=1:FOLDS
        all=[];
        for j=1:2
            Gi=((i-1)*Gs{j})+1;Gj=Gi+Gs{j};
            
            %We are allowing balance groups across cohorts by adding in this bit.
            %Basically if it overshoots the max size for a cohort, it will
            %sample backwards (i.e. resample some of the previous). This
            %should be okay as it is expected that the k-folds will be
            %randomised over many,many iterations. It can be prevented by
            %calculating the right K for your dataset or modifying the code
            %to just ceiling the groups and not resample.
            
            if Gj>sG(j);diff=Gj-sG(j);Gi=Gi-diff+1;Gj=sG(j);end
            %if i==FOLDS;Gj=sG(j);end
            tmp=(GP(G==j));tmp=tmp(:);all=[all;tmp(Go{j}(Gi:Gj))];
        end
        TEST{i}=all;TRAIN{i}=GP;TRAIN{i}(all)=[];
    end
    
else %Now the more complicated multiple group problem
    if size(MERGE(:),1)>2
        disp('Warning - Merge grouping >2. Check results')
    end
    
    %First create the merged group table
    for i=1:size(MERGE(:),1)
        gop=MERGE{i}(:);
        for j=1:size(gop,1)
            G(GROUPS==gop(j))=i;
        end
        msG(i)=sum(G==i);% Add in this bit to pool accross groups
    end
    %Quick check
    if FOLDS>min(msG)
        disp('Error - More folds than data available');
        TEST=NaN;TRAIN=NaN;LABELS=NaN;
        %Note this will allow for small samples to appear accross folds if
        %in combined group
        return
    end
    
    if FOLDS>min(sG) %Not ideal but permissable, warn users, review results
        disp('Warning - More folds than one of subgroups, but merged okay.');
        disp('-> This will mean replication of sub-data in merged groups accross folds');
        disp('-> This could cause overfitting, so review the results');
        disp('-> You could also try either more data or less folds');
    end
    
    GP=1:1:size(G,1); Gs=cell(1,size(MERGE(:),1));Go=Gs;
    
    %Figure out data proportions
    for j=1:size(uG,1) %accross all data groups
        Gs{j}=round(sG(j)*Ks);Go{j}=randperm(sG(j));
    end
    
    for i=1:FOLDS
        all=[];
        for j=1:size(uG,1)
            Gi=((i-1)*Gs{j})+1;Gj=Gi+Gs{j};
            if Gj>sG(j);diff=Gj-sG(j);Gi=Gi-diff;Gj=sG(j);end
            %if i==FOLDS;Gj=sG(j);end
            tmp=(GP(GROUPS==uG(j)));tmp=tmp(:);all=[all;tmp(Go{j}(Gi:Gj))];
        end
        TEST{i}=all;TRAIN{i}=GP;TRAIN{i}(all)=[];
    end
end
LABELS=G;
end

