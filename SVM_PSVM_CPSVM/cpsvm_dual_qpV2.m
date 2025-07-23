%% Solving the following Quadratic Problems with quadprog 

% Dual problem of CPSVM model (V2):
%
%  minimize    0.5*x'*Q*x + f'*x
%  subject to  A*x=b
%              0<= x <= Cv
% 
% where x=[alpha;beta;gamma] in R^{3m}
%       f=[-0.5*Y-0.5*epsi*e; 0; e] + C2*[D*K*Y;K*Y;-K*Y] in R^{3m}
%       A=[Y', e', -e'] in R^{3m}
%       b=-C2*sum(Y)

% Reference (eq. (38) of [1], which is the dual of eq. (19) of [2]):
% [1] Shao YH, Lv XJ, Huang LW, et al (2023) 
%     Twin svm for conditional probability estimation in binary and multiclass
%     classification. Pattern Recognition 136:109253
% [2] Carrasco, Ivorra, López, et al. (2025)
%     A Study of PSVM and CPSVM Models: Analysis, Correction, and 
%     Application in Operations Research

function [Prediction,Tf,S]=cpsvm_dual_qpV2(X,Y,Xtest,FunPara)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%   X        - Training data matrix (m × n), each row is a data point
%   Y        - Training labels (m × 1), entries in {+1, -1}
%   Xtest    - Test data matrix (d × n), each row is a data point
%
%   FunPara  - Struct with parameters:
%                .C1, .C2 > 0   : Regularization parameters
%                .epsi ∈ (0,1]  : Margin parameter
%                .kerfPara      : Kernel parameters (.type = 'lin' or 'rbf')
%
% Outputs:
%   Prediction - Predicted labels (+1/-1) for Xtest
%   Tf         - CPU time (seconds) for solving QP
%   S          - Struct with:
%                 .alpha, .beta, .gamma: dual variables
%                 .b: bias term
%                 .w (if linear): weight vector
%                 .Prob: decision values for Xtest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract parameters
C1=FunPara.C1;
C2=FunPara.C2;
epsi=FunPara.epsi;
kerfPara = FunPara.kerfPara;

[m, ~] = size(X);
e=ones(m,1);

%% Compute kernel matrix
if strcmp(kerfPara.type,'lin')
    K = X*X';
else
    K=kernelfun(X,kerfPara);
end

% Construct Q matrix
K1 = K .* (Y * Y');     % Y*Y' ∘ K
K2 = K .* Y;            % Y ∘ K
K3 = K * diag(Y);       % K*diag(Y)

Q=[K1,K2,-K2;K3, K, -K; -K3, -K, K];
Q=(Q+Q')/2;           % Ensure symmetry
Q=Q+1.e-8*eye(3*m);   % Regularization

%% Linear term f
f1=[-0.5*Y-0.5*epsi*e; zeros(m,1); e];
Ky=K*Y;
f2=[diag(Y)*Ky;Ky;-Ky];
f=f1+C2*f2;

%% Equality constraint A*x = b
Aeq=[Y', e', -e'];
beq=-C2*sum(Y);

%% Bounds
lb = zeros(3*m,1);
ub=[(C1/epsi)*e;Inf*e;Inf*e];

%% Solve QP
t0=cputime;
options = optimoptions('quadprog', 'Display', 'off');
[sol, ~, ~, ~, lambda]= quadprog(Q,f,[],[],Aeq,beq,lb,ub,[],options); 
Tf=cputime-t0;

%% Extract solution
S.alpha  = sol(1:m);
S.beta   = sol(m+1:2*m);
S.gamma  = sol(2*m+1:3*m);
S.b      = lambda.eqlin;

%% Prediction on test data
if strcmp(kerfPara.type,'lin')
   w=X'*(C2*Y+S.alpha .* Y + S.beta - S.gamma);
   Prob_Xt=Xtest*w + S.b;
   S.w=w;
else
   Kt=kernelfun(X,kerfPara,Xtest);
   Prob_Xt=Kt'*(C2*Y+S.alpha .* Y + S.beta - S.gamma)+S.b;
end

Prediction=sign(Prob_Xt-0.5);
S.Prob=Prob_Xt;

end
