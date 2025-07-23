%% Cross Validation for CPSVM model, version 1 (linear kernel)
%
% Reference (eq. (37)):
% Shao YH, Lv XJ, Huang LW, et al. (2023) 
% Twin svm for conditional probability estimation in binary and multiclass 
% classification. Pattern Recognition 136:109253


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
CV = 10;          % number of folds
Cl = -7;          % lower bound for C exponent
Ch = 7;           % upper bound for C exponent
El = -7;          % Lower bound for epsi exponent
Eh = 0;           % Upper bound for epsi exponent

%% Allocate result matrices (3D: epsi × C1 × C2)
AUCMATRIX  = zeros(Eh-El+1, Ch-Cl+1, Ch-Cl+1);
ACCUMATRIX = zeros(Eh-El+1, Ch-Cl+1, Ch-Cl+1);

FunPara.kerfPara.type = 'lin';

%% Grid search over (epsi, C1, C2)
for e = El:Eh
    fprintf('Epsi exponent: %d\n', e);
    FunPara.epsi = 2^e;

    for i = Cl:Ch
        fprintf('  C1 exponent: %d\n', i);
        FunPara.C1 = 2^i;

        for j = Cl:Ch
            fprintf('    C2 exponent: %d\n', j);
            FunPara.C2 = 2^j;

            for k = 1:CV
                tst = perm(k:CV:m);         % Test indices
                trn = setdiff(1:m, tst);    % Training indices

                % Training data
                Ytr = Y(trn,:);
                Xtr = X(trn, :);
                % Test data
                Yt  = Y(tst',:);
                Xt  = X(tst', :);

                % Train and test CPSVM v1
                [prediction, Tf] = cpsvm_dual_qpV1(Xtr, Ytr, Xt, FunPara);
                [AUC(k), Accu(k)] = medi_auc_accu(prediction, Yt);
            end

            AUCMATRIX(e-El+1, i-Cl+1, j-Cl+1)  = mean(AUC);
            ACCUMATRIX(e-El+1, i-Cl+1, j-Cl+1) = mean(Accu);
        end
    end
end

%% Save results to .mat file
Out_results = struct();
Out_results.AUCMATRIX   = AUCMATRIX;
Out_results.ACCUMATRIX  = ACCUMATRIX;
Out_results.C1_range    = Cl:Ch;
Out_results.C2_range    = Cl:Ch;
Out_results.epsi_range  = El:Eh;
Out_results.CV          = CV;

save('CPSVM_v1_CV_results_with_epsi.mat', 'Out_results');

disp('Results saved to CPSVM_v1_CV_results_with_epsi.mat');

