%% Cross Validation for PSVM (with rbf kernel)
%
% Grid search over epsi, C, and sigma parameters
% Evaluated using k-fold cross-validation

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
Sil = -7;          % lower bound for sigma exponent
Sih = 7;           % upper bound for sigma exponent
El = -7;          % Lower bound for epsi exponent
Eh = 0;           % Upper bound for epsi exponent

FunPara.kerfPara.type = 'rbf';  % use RBF kernel

%% Allocate result matrices (3D: epsi × C × sigma)
AUCMATRIX  = zeros(Eh-El+1, Ch-Cl+1, Sih-Sil+1);
ACCUMATRIX = zeros(Eh-El+1, Ch-Cl+1, Sih-Sil+1);

%% Grid search over (epsi, C, sigma)
for e = El:Eh
    fprintf('Epsi exponent: %d\n', e);
    FunPara.epsi = 2^e;

    for i=Cl:Ch
        fprintf('  C exponent: %d\n', i);
        FunPara.C =2^i;
        for j=Sil:Sih
            fprintf('  Sigma exponent: %d\n', j);
            FunPara.kerfPara.pars=2^j;

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
                [prediction] = PSVM_qp(Xtr, Ytr, Xt, FunPara);
                [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
            end
            AUCMATRIX(e-El+1,i-Cl+1,j-Sil+1)=mean(AUC);
            ACCUMATRIX(e-El+1,i-Cl+1,j-Sil+1)=mean(Accu);
        end
    end
end

%% Save results to .mat file
Out_results = struct();
Out_results.AUCMATRIX   = AUCMATRIX;
Out_results.ACCUMATRIX  = ACCUMATRIX;
Out_results.epsi_range  = El:Eh;
Out_results.C_range     = Cl:Ch;
Out_results.sigma_range = Sil:Sih;
Out_results.CV          = CV;
Out_results.kernel      = 'rbf';

save('PSVM_CV_results_kernel.mat', 'Out_results');

disp('Results saved to PSVM_CV_results_kernel.mat');

