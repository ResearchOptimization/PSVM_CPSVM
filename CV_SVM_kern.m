%% Cross Validation for SVM soft margin (nonlinear kernel)

clc
clear all

addpath(genpath('./dataset_bin'))
addpath(genpath('./SVM_PSVM_CPSVM'))

%% Load dataset

load 'sonar.mat'
%load('heart_statlogN.mat');
%load('bupa_liverN.mat');


[m, n]=size(X);

%% Cross-validation and parameter grid
folds = 10;          % number of folds
Cl = -7;          % lower bound for C exponent
Ch = 7;           % upper bound for C exponent
Sil = -7;          % lower bound for sigma exponent
Sih = 7;           % upper bound for sigma exponent

FunPara.kerfPara.type = 'rbf';

%% Allocate result matrices (C × sigma)

AUCMATRIX=zeros(Ch-Cl+1);
ACCUMATRIX=zeros(Ch-Cl+1);

for i=Cl:Ch
    fprintf('  C exponent: %d\n', i)
    FunPara.c=2^i;
    for j=Sil:Sih
        fprintf('  Sigma exponent: %d\n', j)
        FunPara.kerfPara.pars = 2^j;
        for k=1:folds
            tst=perm(k:folds:m); % test indices
            trn=setdiff(1:m,tst); % training indices
            % Training data
            Ytr=Y(trn,:);
            Xtr=X(trn,:);
            % Test data
            Yt=Y(tst',:);
            Xt=X(tst',:);
             % Train and test SVM  
            prediction = SVM_soft_quadsolve(Xtr,Ytr,Xt,FunPara); % via Quadolve
            [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
        end
        AUCMATRIX(i-Cl+1,j-Cl+1)=mean(AUC);
        ACCUMATRIX(i-Cl+1,j-Cl+1)=mean(Accu);
    end
end

Out_results = struct();
Out_results.AUCMATRIX  = AUCMATRIX;
Out_results.ACCUMATRIX = ACCUMATRIX;
Out_results.C_range     = Cl:Ch;
Out_results.Sigma_range   = Sil:Sih;
Out_results.folds       = folds;

%% Save results to .mat file
save('SVM_soft_kern_CV_results.mat','Out_results');

disp('Results saved to SVM_soft_kern_CV_results.mat');
