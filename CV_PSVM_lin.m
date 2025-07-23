%% Cross Validation for PSVM (linear kernel)
%

clc
clear all

addpath(genpath('./dataset_bin'))
addpath(genpath('./SVM_PSVM_CPSVM'))

%% Load dataset
load 'sonar.mat'
%load('heart_statlogN.mat');
%load('bupa_liverN.mat');


[m,n]=size(X);

%% Cross-validation and parameter grid
CV = 10;          % number of folds
Cl = -7;          % lower bound for C exponent
Ch = 7;           % upper bound for C exponent
Ceps_l = -3;      % lower bound for epsilon exponent
Ceps_h = 0;       % upper bound for epsilon exponent

%% Allocate result matrices (epsi × C)
AUCMATRIX=zeros(Ceps_h-Ceps_l+1,Ch-Cl+1);
ACCUMATRIX=zeros(Ceps_h-Ceps_l+1,Ch-Cl+1);

FunPara.kerfPara.type = 'lin';

for i=Ceps_l: Ceps_h
    fprintf('Epsilon exponent: %d\n', i)
    FunPara.epsi = 2^i;   % note: 'eps' is reserved in MATLAB, use 'epsi'
    
    for j=Cl:Ch
        fprintf('  C exponent: %d\n', j)
        FunPara.C = 2^j;
        for k=1:CV
            tst=perm(k:10:m);      % Test indices
            trn=setdiff(1:m,tst);  % Training indices

            % Training data
            Ytr=Y(trn,:);
            Xtr=X(trn,:);
            % Test data
            Yt=Y(tst',:);
            Xt=X(tst',:);
            % Train and test PSVM  
            [prediction,Tf] = PSVM_qp(Xtr, Ytr, Xt, FunPara);
            [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
        end
        AUCMATRIX(i-Ceps_l+1,j-Cl+1)=mean(AUC);
        ACCUMATRIX(i-Ceps_l+1,j-Cl+1)=mean(Accu);
    end
end

Out_results = struct();
Out_results.AUCMATRIX  = AUCMATRIX;
Out_results.ACCUMATRIX = ACCUMATRIX;
Out_results.C_range     = Cl:Ch;
Out_results.epsi_range  = Ceps_l:Ceps_h;
Out_results.CV          = CV;

%% Save results to .mat file
save('PSVM_CV_results.mat', 'Out_results');

disp('Results saved to PSVM_CV_results.mat');
