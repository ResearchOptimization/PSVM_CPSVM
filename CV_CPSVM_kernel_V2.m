%% Cross Validation for CPSVM model, version 2 (with RBF kernel)
%
% Grid search over (epsi, C1, C2, sigma) with k-fold cross-validation
%

clc
clear all

addpath(genpath('./dataset_bin'))
addpath(genpath('./SVM_PSVM_CPSVM'))

%% Load dataset
load 'sonar.mat'
%load('heart_statlogN.mat');
%load('bupa_liverN.mat');


[m, n] = size(X);
CV = 10;             % number of folds

%% Parameter ranges
Cl = -7;             % lower bound for C1 and C2 exponents
Ch = 7;              % upper bound for C1 and C2 exponents
Sl = -7;             % lower bound for sigma exponent
Sh = 7;              % upper bound for sigma exponent
El = -7;             % lower bound for epsilon exponent
Eh = 0;              % upper bound for epsilon exponent

FunPara.kerfPara.type = 'rbf';

%% Loop over epsi
for indm = El:Eh
    fprintf('\n=== Starting epsi exponent: %d ===\n', indm);
    FunPara.epsi = 2^indm;

    % Allocate results for current epsi
    AUCMATRIX  = zeros(Ch-Cl+1, Sh-Sl+1);
    ACCUMATRIX = zeros(Ch-Cl+1, Sh-Sl+1);

    t0 = cputime;

    % Grid search over C1, C2, sigma
    for i = Cl:Ch
        fprintf('  C1 & C2 exponent: %d\n', i);
        FunPara.C1 = 2^i;
        FunPara.C2 = 2^i;   % here C1 = C2

        for j = Sl:Sh
            fprintf('    Sigma exponent: %d\n', j);
            FunPara.kerfPara.pars = 2^j;

            for k = 1:CV
                tst = perm(k:CV:m);      % test indices
                trn = setdiff(1:m, tst); % training indices

                % Training data
                Ytr = Y(trn,:);
                Xtr = X(trn,:);

                % Test data
                Yt  = Y(tst',:);
                Xt  = X(tst',:);

                % Train and test CPSVM (v1)
                [Prediction, Tf] = cpsvm_dual_qpV2(Xtr, Ytr, Xt, FunPara);
                [AUC(k), Accu(k)] = medi_auc_accu(Prediction, Yt);
            end

            AUCMATRIX(i-Cl+1, j-Sl+1)  = mean(AUC);
            ACCUMATRIX(i-Cl+1, j-Sl+1) = mean(Accu);
        end
    end

    elapsedTime = cputime - t0;
    fprintf('Elapsed CPU time for epsi=%g: %.2f seconds\n', FunPara.epsi, elapsedTime);

    %% Save results for this epsi
    Out_results = struct();
    Out_results.AUCMATRIX   = AUCMATRIX;
    Out_results.ACCUMATRIX  = ACCUMATRIX;
    Out_results.C1_C2_range = Cl:Ch;
    Out_results.sigma_range = Sl:Sh;
    Out_results.epsi        = FunPara.epsi;
    Out_results.CV          = CV;
    Out_results.kernel      = 'rbf';
    Out_results.elapsedTime = elapsedTime;

    filename = sprintf('CPSVM_v2_kern_CV_results_epsi_%d.mat', indm);
    save(filename, 'Out_results');
    fprintf('Results saved to %s\n', filename);
end

