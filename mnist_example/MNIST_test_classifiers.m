%% MNIST classification with a Gaussian Process model and an SVM 

% Data: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
% Read MNIST script from https://uk.mathworks.com/matlabcentral/fileexchange/27675-read-digits-and-labels-from-mnist-database

%Load data 
[imgs, labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 10000, 0); 
%% 
%Select characters to classify for simplicity use 1 and another digit 
idx = labels == 1 | labels == 4 ;
filter_labels = labels(idx) ; filter_imgs = imgs(:,:,idx) ;
n_samples = length(filter_labels);

%Now change to +1, -1 for sake of classifier 
%idx_2 = filter_labels == 2 ; filter_labels(idx_2) = -1 ;

%Check images and labels  are as expected with 1 and -1 labels correct
%imshow(filter_imgs(:,:,1)) ; filter_labels(1)

%Now flatten into vector ;
flat_imgs = reshape(permute(filter_imgs, [3 1 2]), [], 400) ; 

%Split into train and test set - fine for now but check for class balance
%etc. and CV over hyperparams later

train_x = flat_imgs(1:0.8*n_samples ,:) ;
test_x = flat_imgs(0.8*n_samples:end ,:) ; 
train_y = filter_labels(1:0.8*n_samples ,:) ;
test_y = filter_labels(0.8*n_samples:end ,:) ;

%% Now test some classifiers 

%% 01 SVM for baseline 
SVMModel = fitcsvm(train_x,train_y,'KernelFunction','linear',...
    'Standardize',true,'ClassNames',[-1,1]);

ScoreSVMModel = fitPosterior(SVMModel, train_x,train_y);

%scores
[predict_y,score] = predict(ScoreSVMModel,test_x);

% accuracy calc getting 98.6 ish FOR 1 VS 8   
SVM_acc = 100*(1 - sum(predict_y ~= test_y) / n_samples) ;

%Can plot some hard examples... 

error_idx_SVM = find(predict_y ~= test_y); 


%% Feature analysis - which features are important for discrimination?
%This isn't the full recursive feature selection routine, just rough
%illustration 

idx_support = find(SVMModel.IsSupportVector);
support_data = train_x(idx_support ,:);
%multiply by alpha
feature_SVM = SVMModel.Alpha'*support_data;
figure;
imshow(reshape(feature_SVM, 20, 20)); 

%% 02 Gaussian Process model 
%See binaryLaplaceGP for more information on this... also p.4 of the GP
%toolbox manual 

%See GP manual p6 for more detail 
% Usage: 
% training
% [nlZ dnlZ          ] = gp(hyp, inf, mean, cov, lik, x, y);
% prediction
% [ymu ys2 fmu fs2   ] = gp(hyp, inf, mean, cov, lik, x, y, xs);
% 
%    or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, mean, cov, lik, x, y, xs, ys);
% 
% hyp = struct of column vectors of mean/cov/lik hyperparameters
% inf = function specifying the inference method
% mean = prior mean function
% cov = prior covariance function
% lik = likelihood function
% x = n by D matrix of training inputs
% y = column vector of length n of training targets
% xs = ns by D matrix of test inputs
% ys = column vector of length nn of test targets
% 
% nLz = negative log marginal likelihood 
% dnLz = partial derivatives of nLz wrt mean/cov/like hyperparams
% ymu = column vector of predicted output means 
% ys2 = column vector of predicted output variances 
% lp = column vector of log predicted probabilities 
% post = struct representation of the (approximate) posterior 3rd output in training mode 
%         6th output in prediction mode 
%         
% Depending on the number of input parameters, 
% gp knows whether it is operated in training or in prediction mode.

%mean of zero as targets +1 -1 ?correct interpretation 
mean = {@meanZero} ; 
%Using a linear covariance function for now - no associated hyps
cov = {@covMatern}; 
% Using Laplace for approximation of the posterior for now (EP slightly
% better but much slower on this dataset thus far) 
inf = {@infLaplace};
% 'Likelihood function' see page 40 rassmussen, p.20 toolbox guide
%This is the function we use to generate our prediction from the latent
%function outputs i.e. map to between 0 and 1 - e.g. sigmoid,
lik = {@likLogistic}; 
%note that inputting just the lik gives no. of hyperparameters it expects 
            
%hyperparameters for each of the above 
hyp.mean = [] ;
hyp.cov = [] ;
hyp.lik = [] ;

%% Run model 
[ymu, ys2, fmu, fs2  ,~, post ] = gp(hyp, inf, mean, cov, lik, train_x, train_y, test_x);

%ymu is the predictive mean see p 20 manual 
% ymu =  integral{ y*  p(y* | x*, D) }dy* 
% so we can predict the class using the sign of this output  

%ys2 is the predictive variance 
% ys2 =  integral{ (y* - ymu)^2  p(y* | x*, D) }dy* 
GP_acc = 100*(1 - sum(sign(ymu) ~= test_y) / n_samples) ;

%% Visualise some results 
figure; 
subplot(2, 1, 1)
%histogram of the latent means
histogram(fmu, 100);
title('Latent means: fmu') 

subplot(2, 1, 2);
%histogram of the predictions
histogram(ymu, 100);
title('Predictive  ymu') 
hold off; 

% Below we investigate the findings with
%1. Qualitative review of errors 
%2. Review of means/variances of error examples 
%3. Feature analysis - which pixels are particularly predictive ? 
% TODO - test some different GP settings - keeping it linear for now 

%% Qualitative review
%Invert our original transformation for plotting images 

original_shapes = permute(reshape(test_x, length(test_x), 20, 20), [2 3 1]) ;

error_idx = find(sign(ymu) ~= test_y); 

figure;
subplot(2,2,1)
imshow(original_shapes(:,:,error_idx(1)));
title(sprintf('Classified as %d', sign(ymu(error_idx(1))))) 

subplot(2,2,2)
imshow(original_shapes(:,:,error_idx(2)));
title(sprintf('Classified as %d', sign(ymu(error_idx(2))))) 

subplot(2,2,3)
imshow(original_shapes(:,:,error_idx(3)));
title(sprintf('Classified as %d', sign(ymu(error_idx(3))))) 
subplot(2,2,4)
imshow(original_shapes(:,:,error_idx(4)));
title(sprintf('Classified as %d', sign(ymu(error_idx(4))))) 
    
%% Quantitative - what outputs are we getting for each error example? 

%Means of predictions
figure;
histogram(ymu, 100) ;
hold on
for i=1:numel(error_idx)
    plot(ymu(error_idx(i)), 0, 'r*')
end
title('Means of predictions with error examples *') 

%Variances of predictions
figure;
histogram(ys2, 100) ;
hold on
for i=1:numel(error_idx)
    plot(ys2(error_idx(i)), 0, 'r*')
end
title('Variance of predictions with error examples *') 

%% Feature importance 
%Note the full recursive routine will be implemented for the neuroimaging
%data

%Approach to feature analysis here will be
% 1. Identify important training examples - importance denoted by the abs(weight) of the kernel for that sample
% 2. Importance*feature map  summed for all samples 
% -> 20x20 map of features weighted by importance
%Re-read p44 GP textbook for interpretation of coefficients... 

%Recap on the theory to understand where the alpha comes in! 

%We're using the GP to model a latent function f(x) 
%We pass this through a sigmoid function sig(f(x)) to obtain a class
%probability 
%This class probability is weighted by the prob of that function 

%We integrate over all possible latent functions f when computing this
% --> the posterior p(f | X, y)  is non-gaussian thus gives an intractable integral 

%We have approximated the posterior (e.g. with Laplace approx)
% See inf methods for detail - but the approximated posterior to be
% returned is in the form N(mu=m+K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal

%TBC - my interpretation is that the output mean is a function of x_train with alpha is the weight for 
%each kernel output
%distribution over alphas
histogram(post.alpha) ; 

%% Plot the most discriminatory training examples and some non-discriminatory examples

idx_max = find(post.alpha == max(post.alpha)) ;
idx_min = find(post.alpha == min(post.alpha)) ;

%Manually found some small ~0 alphas by inspecting post.alpha 
idx_small1 = 3 ; %check these each run!
idx_small2 = 1 ; %check these each run!
%idx_small3 = 116 ;

original_train = permute(reshape(train_x, length(train_x), 20, 20), [2 3 1]) ;

figure;
subplot(2, 2, 1);
imshow(original_train(:,:,idx_max));
title(sprintf('alpha = %d', max(post.alpha))) 
subplot(2, 2, 2);
imshow(original_train(:,:,idx_min));
title(sprintf('alpha = %d', min(post.alpha))) 
subplot(2, 2, 3);
imshow(original_train(:,:,idx_small1));
title(sprintf('alpha = %d', post.alpha(idx_small1))) ;
subplot(2, 2, 4);
imshow(original_train(:,:,idx_small2));
title(sprintf('alpha = %d', post.alpha(idx_small2))) ;

%% Show a 'feature map' of the most discriminatory areas 

features = abs(post.alpha)'*train_x ;
figure;
imshow(reshape(features, 20, 20));

%Quite messy - GP is not a sparse classifier... 

%now try with only extreme alphas...
%abs(post.alpha(abs(post.alpha) > 0.1);
%%
extr_alpha = abs(post.alpha) > 0.15;
extr_features = abs(post.alpha(extr_alpha))'*train_x(extr_alpha,:);
figure;
imshow(reshape(extr_features, 20, 20)); 

%% Linear and Quadratic Discriminant Analyses testing examples


%% sample found online https://stackoverflow.com/questions/48504297/linear-discriminant-analysis-matlab
x = [1, 1.5] .* randn(100, 2);
x(51:end, :) = [1, 2] .* x(51:end, :) + [2, 4];
y = [ones(50, 1); 2*ones(50, 1)];

mdl = fitcdiscr(x, y, 'DiscrimType', 'linear');
%% Plotting for LDA 

figure(1)
gscatter(x(:, 1), x(:, 2), y);
hold on 

lx = get(gca, 'Xlim');

%Note on plot 
% Const + Linear * x = 0 in the LDA case 
%Can calculate the function of the line - one domain as a function of
%another - NOT y (label) as a function of x - we'e plotting a line through
%the feature space. Rearrange the equation. 

%So can plot x(2) = -(Const + Linear(1) * x(1)) / Linear(2)
ly = -(mdl.Coeffs(1, 2).Const + mdl.Coeffs(1, 2).Linear(1) .* lx) / mdl.Coeffs(1, 2).Linear(2);
plot(lx, ly, '-b', 'DisplayName', 'LDA')

hold off

%% Quadratic Equivalent

x = [1, 1.5] .* randn(100, 2);
x(51:end, :) = [1, 2] .* x(51:end, :) + [2, 4];
y = [ones(50, 1); 2*ones(50, 1)];

mdl = fitcdiscr(x, y, 'DiscrimType', 'Quadratic');

k = mdl.Coeffs(1,2).Const;
l = mdl.Coeffs(1,2).Linear;
q = mdl.Coeffs(1,2).Quadratic;
f = sprintf('0 = %g+%g*x+%g*y+%g*x^2+%g*x.*y+%g*y.^2', k, l, q(1,1), q(1,2)+q(2,1), q(2,2));

% https://uk.mathworks.com/help/stats/classify.html 
%f = @(x,y) K + L(1)*x + L(2)*y + Q(1,1)*x.*x + (Q(1,2)+Q(2,1))*x.*y + Q(2,2)*y.*y;
figure(1)
gscatter(x(:, 1), x(:, 2), y);
hold on 

ezplot(f, [min(x(:,1)), max(x(:,1)), min(x(:,2)), max(x(:,2))])

hold off

%% Linear discrim on MNIST 

mdl = fitcdiscr(train_x, train_y, 'DiscrimType', 'Linear');

lin_disc_acc = 100*(1 - sum(mdl.predict(test_x) ~= test_y) / n_samples) ;

%% Quadratic errors when trying to get working ! 

train_x_tab = array2table(train_x) ;

mdl = fitcdiscr(train_x_tab, train_y, 'DiscrimType', 'Linear');

%Predictor train_x2 has zero variance for class 1. Either exclude this predictor or
%set 'discrimType' to 'pseudoQuadratic' or 'diagQuadratic'.

%Have tried to transform/alter things but not reaaaly working... 
%x_transp = train_x'; 
% mod_x = train_x ;
% mod_y = train_y;
% mod_x(2,:) = [] ;
% mod_y(2) = [] ;

%error_idx_lin = find(mdl.predict(test_x) ~= test_y); 

%Unsure as the variance of the train data seems fine...
variance = var(train_x, 1, 2) ;

%Issue is that the covariance matrix for class 1 is singular!!!!! 
check = train_x(find(train_y == 1),:) ;

rank(check) % = 322! but 400 cols thus not able to find an inverse for this

%For quadratic iscriminant we need to invert the cov matrices for each
%class which isn't possible in this way !! 

%% Testing out PCA on the data 

[coeff, scores, latent] = pca(train_x, "NumComponents", 14) ; 

transform_testx = test_x*coeff;
% 
mdl = fitcdiscr(scores, train_y, 'DiscrimType', 'Quadratic') ;

lin_disc_acc = 100*(1 - sum(mdl.predict(trasnform_testx) ~= test_y) / n_samples) ;
% 

%Degree of regularization impacts
%performance:%https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6869036/  
%%

%Plotting model performance as a function of PCA components 
PCA_comps = 1:20;
Model_acc = zeros(20, 1);

for i=1:20
    
    [coeff, scores, latent] = pca(train_x, "NumComponents", i) ; 
    transform_testx = test_x*coeff; 
    
    mdl = fitcdiscr(scores, train_y, 'DiscrimType', 'Quadratic') ;
    lin_disc_acc = 100*(1 - sum(mdl.predict(transform_testx) ~= test_y) / n_samples) ;
    Model_acc(i) = lin_disc_acc;
    
end

figure;
plot(PCA_comps, Model_acc);
title('Linear Disc. Accuracy as a function of model PCA components');

%% Playing with LDA - can we use this as input to QDA? ?do we need to? 

%Kernelize it to test out... 

XK = train_x*train_x' ;

mdl = fitcdiscr(XK, train_y, 'DiscrimType', 'Linear') ;

testXK = test_x*train_x' ; 

lin_disc_acc = 100*(1 - sum(mdl.predict(testXK) ~= test_y)/n_samples) ;

%% Try the Quadratic Form of this... 

XK = train_x*train_x' ;

mdl = fitcdiscr(XK, train_y, 'DiscrimType', 'Quadratic') ;

testXK = test_x*train_x' ; 

lin_disc_acc = 100*(1 - sum(mdl.predict(testXK) ~= test_y)/n_samples) ;

%% Check Covariance Matrix of the Kernel 

%Check the covariance of the class 1
check = XK(find(train_y == -1),:) ;

rank(check) % = 322! but 400 cols thus not able to find an inverse for this

%QQQQ- check this.... Why is the rank of the kernel the same as that of the original feature
%space? 



%% %Plotting model performance as a function of PCA components with kernelized features
PCA_comps = 1:20;
Model_acc = zeros(20, 1);

for i=1:20
    
    [coeff, scores, latent] = pca(XK, "NumComponents", i) ; 
    %kernelised test set 
    
    transform_testx = testXK*coeff;
    
    mdl = fitcdiscr(scores, train_y, 'DiscrimType', 'linear') ;
    lin_disc_acc = 100*(1 - sum(mdl.predict(transform_testx) ~= test_y) / n_samples) ;
    Model_acc(i) = lin_disc_acc;
    
end

% plot 
figure;
plot(PCA_comps, Model_acc);
title('Linear Disc. Accuracy as a function of model PCA components');

%% Feature selection/examination 

%Rationale for this approach
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

%If model not run earlier ----------
%Generate the kernel Matrices 
XK = train_x*train_x' ;
testXK = test_x*train_x' ; 

%Learn Principle Components 
[coeff, scores, latent] = pca(XK, "NumComponents", 4) ; 

%Transform test data 
transform_testx = testXK*coeff;

%Fit model 
mdl = fitcdiscr(scores, train_y, 'DiscrimType', 'linear') ;
lin_disc_acc = 100*(1 - sum(mdl.predict(transform_testx) ~= test_y) / n_samples) ;

%Norm of the weighted training data in PC space 
subject_weight = abs(scores)*abs(mdl.Coeffs(1, 2).Linear) ;

%Weight the original data by it's importance in classification
features = subject_weight'*train_x ;

%Normalize for plotting purposes...
norm_features = mat2gray(features);
figure;
imshow(reshape(norm_features, 20, 20));

%% 





