function accuracy=SVMbin_METRICS(LABELS,TARGET)
%Input the output from SVM labels. Two class problem (1 or 2)
%Output BA, sens, spec, PPV, NPV
%--------------------------------------------------------------------------
% C Lambert - 
% Version 1.0 - July 2017
% Version 1.1 - June 2021
%--------------------------------------------------------------------------

labs=unique(TARGET(:));

if size(labs,1)~=2
    disp('Error - This function is for a two class problem');
    return
else
    %Work out table
    TPF=sum((TARGET(LABELS==labs(2)))==labs(2));TNF=sum((TARGET(LABELS==labs(1)))==labs(1));
    FPF=sum((TARGET(LABELS==labs(2)))==labs(1));FNF=sum((TARGET(LABELS==labs(1)))==labs(2));
    
    %Now get metrics
    accuracy.acc=sum(TARGET~=LABELS)/size(TARGET,1);
    accuracy.ba= 0.5* ((TPF/(TPF+FNF))+(TNF/(FPF+TNF)));
    accuracy.sens=TPF/(TPF+FNF);
    accuracy.spec=TNF/(TNF+FPF);
    accuracy.ppv=TPF/(TPF+FPF);
    accuracy.npv=TNF/(TNF+FNF);
    accuracy.tp=TPF;
    accuracy.tn=TNF;
    accuracy.fp=FPF;
    accuracy.fn=FNF;
end
end