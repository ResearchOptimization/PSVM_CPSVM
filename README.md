
 Title: A Study of PSVM and CPSVM Models: Analysis, Correction, and Application in Operations Research
 =============================================================================

 Overview:
 This project implements Cross-Validation (CV) schemes for binary classification 
 using the following models:
   - SVM
   - PSVM
   - CPSVM (version 1)
   - CPSVM (version 2)

 Folder Structure:
 -----------------
 SVM/                - Contains optimization code for SVM, PSVM, and CPSVM models:
     kernelfun.m
     quadsolve.m
     SVM_soft_quadsolve.m
     PSVM_qp.m
     cpsvm_dual_qpV1.m
     cpsvm_dual_qpV2.m

 dataset_bin/        - Contains benchmark datasets in .mat format:
     bupa_liver.mat
     heart_statlog.mat
     sonar.mat

 Main Components:
 ----------------
 1. Datasets (dataset_bin/):
    - Sample datasets for evaluation.

 2. CV Programs:
    - Linear kernel:
       CV_SVM_lin.m          : CV for SVM
       CV_PSVM_lin.m         : CV for PSVM
       CV_CPSVM_V1.m         : CV for CPSVM (v1)
       CV_CPSVM_V2.m         : CV for CPSVM (v2)

    - Nonlinear kernel:
       CV_SVM_kern.m         : CV for SVM
       CV_PSVM_kernel.m      : CV for PSVM
       CV_CPSVM_kernel_V1.m  : CV for CPSVM (v1)
       CV_CPSVM_kernel_V2.m  : CV for CPSVM (v2)

 3. Evaluation Metrics:
    - medi_auc_accu.m        : Computes accuracy and balanced accuracy

 Usage:
 ------
 Step 1: Configure parameters in the desired CV_*.m file
         - 'case'  : Dataset name (e.g., 'sonar')
         - 'folds' : Number of CV folds (default: 10)

 Step 2: Run the script in MATLAB:
         - Loads data, trains models, evaluates via CV
         - Outputs metrics: Accuracy, Balanced Accuracy

 Step 3: Inspect results:
         - ACCUMATRIX : Accuracy matrix for each hyperparameter
         - AUCMATRIX  : Balanced Accuracy matrix for each hyperparameter

 Requirements:
 -------------
 - MATLAB (R2021b or later)

 Examples:
 ---------
 Example 1: Run linear SVM on 'sonar.mat'
   - Set dataset in CV_SVM_lin.m
   - Run script

 Example 2: Run nonlinear PSVM on 'sonar.mat'
   - Set dataset in CV_PSVM_kernel.m
   - Run script

 Contact:
 --------
 - Miguel Carrasco   : macarrasco@miuandes.cl
 - Benjamin Ivorra   : ivorra@ucm.es
 - Julio López       : julio.lopez@udp.cl
