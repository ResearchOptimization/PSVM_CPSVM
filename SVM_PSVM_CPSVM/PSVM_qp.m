% Solving the following Quadratic Problem with quadprog

% Dual problem of PSVM model:
%  minimize    0.5*x'*Q*x + f'*x
%  subject to  A*x=0
%              0<= x <= Cv
% 
% where x=[alpha,beta,gamma] in R^{3m}
%       f=[-0.5*Y'-0.5*epsi*e', 0, e'] in R^{3m}
%       A=[Y', e', -e'] in R^{3m}

% Reference:
% Vapnik V, Izmailov R (2021) Reinforced svm method and memorization mechanisms.
% Pattern Recognition 119:108018

function [Prediction,Tf,S]=PSVM_qp(X,Y,Xtest,FunPara)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%   X        - Training data matrix (m × n), each row is a data point
%   Y        - Training labels (m × 1), with entries in {+1, -1}
%   Xtest    - Test data matrix (d × n), each row is a data point
%   FunPara  - Struct with parameters:
%               .C     > 0   : Regularization parameter
%               .epsi ∈ (0,1]: Parameter to tune
%               .kerfPara    : Kernel parameters:
%                                .type = 'lin' or 'rbf'

% Outputs:
%   Prediction - Predicted labels for Xtest
%   Tf         - CPU time (seconds) to solve QP
%   S          - Struct containing:
%                 .alpha, .beta, .gamma: dual variables
%                 .b: bias term
%                 .w (if linear kernel): weight vector
%                 .Prob: decision values for Xtest

% Extract parameters

C=FunPara.C;
epsi=FunPara.epsi;
kerfPara = FunPara.kerfPara;

[m, ~] = size(X);
e=ones(m,1);

%% Compute Kernel matrix
if strcmp(kerfPara.type,'lin')
    K = X*X';
else
    K=kernelfun(X,kerfPara);
end

% Construct Q matrix
K1=K.*(Y*Y');
K2=K.*Y;      %D*K
K3=K*diag(Y); % K*D

Q=[K1,K2,-K2; K3, K, -K; -K3, -K, K];
Q=(Q+Q')/2;           % Ensure symmetry
Q=Q+1.e-8*eye(3*m);   % Regularize for numerical stability

%% Construct linear term and constraints
f=[-0.5*Y-0.5*epsi*e; zeros(m,1); e];
Aeq=[Y', e', -e'];     % Equality constraint A*x=0
beq = 0;

lb=zeros(3*m,1);             % Lower bounds: 0
ub=[(C/epsi)*e;Inf*e;Inf*e]; % Upper bounds

%% Solve QP
t0=cputime;
options = optimoptions('quadprog', 'Display', 'off');
[sol,~,~,~,lambda]= quadprog(Q,f,[],[],Aeq,beq,lb,ub,[],options); 
Tf=cputime-t0;

%% Extract solution
S.alpha=sol(1:m);
S.beta=sol(m+1:2*m);
S.gamma=sol(2*m+1:3*m);
S.b=lambda.eqlin;

%% Compute prediction for test data
if strcmp(kerfPara.type,'lin')
   w=X'*(S.alpha .* Y + S.beta - S.gamma);
   Prob_Xt=Xtest*w + S.b;
   S.w=w;
else
   Kt=kernelfun(X,kerfPara,Xtest);
   Prob_Xt=Kt'*(S.alpha .* Y + S.beta - S.gamma)+S.b;
end

Prediction=sign(Prob_Xt-0.5);
S.Prob=Prob_Xt;



