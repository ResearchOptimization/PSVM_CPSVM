%% Solving the following Quadratic Problems with quadsolve 

% Dual problem of the classical SVM (soft margin)
%  minimize    0.5*x'*K*x -e'*x
%  subject to  Y*x=0
%              0<= x <= C

% where:
%   x ∈ ℝ^m (dual variables)
%   K = (Y * Y') ∘ Kernel(X, X)


function [Ytest,Tf,Sol]=SVM_soft_quadsolve(X,Y,Xt,FunPara)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%   X        - Training data matrix (m × n), each row is a data point
%   Y        - Training labels (m × 1), entries in {+1, -1}
%   Xt       - Test data matrix (d × n), each row is a data point
%
%   FunPara  - Struct with parameters:
%                .c          > 0   : Regularization parameter C
%                .kerfPara         : Kernel parameters:
%                                      .type = 'lin' or 'rbf'
%
% Outputs:
%   Ytest    - Predicted labels (+1/-1) for Xt
%   Tf       - CPU time (seconds) for solving the QP
%   Sol      - Struct with:
%                .alpha     : Optimal dual variables
%                .b         : Bias term
%                .w         : Weight vector (if linear kernel)
%                .Val_Xt    : Decision function values for Xt
%
% Example:
%   A = rand(50,10);
%   B = rand(60,10);
%   X = [A;B];
%   Y = [ones(50,1); -ones(60,1)];
%   TestX = rand(20,10);
%   FunPara.kerfPara.type = 'lin';
%   FunPara.c = 4;
%
%   [Ytest, Tf, Sol] = SVM_soft_quadsolve(X, Y, TestX, FunPara);

C=FunPara.c;
kerfPara = FunPara.kerfPara;

m = size(X,1);

%% Compute kernel matrix
if strcmp(kerfPara.type, 'lin')
    K = X * X';
else
    K = kernelfun(X, kerfPara);
end

% Adjust with labels: K_ij = y_i y_j K(x_i, x_j)
K = K .* (Y * Y');

%% Solve QP
t0=cputime;
[alpha,bias]= quadsolve(K,-ones(m,1),Y',0,C); 
Tf=cputime-t0;
clear K

%% Process solution
Sol.alpha = alpha;
Sol.b = -bias;   % the output from quadsolve is -b in the primal

alpha_scaled = alpha .* Y;

if strcmp(kerfPara.type, 'lin')
    % Linear kernel: compute weight vector explicitly
    w = X' * alpha_scaled;
    Sol.w = w;
    Sol.Val_Xt = Xt * w - bias;
else
    % Non-linear: use kernel on test data
    Kt = kernelfun(X, kerfPara, Xt);
    Sol.Val_Xt = Kt' * alpha_scaled - bias;
end

Ytest=sign(Sol.Val_Xt);

end
