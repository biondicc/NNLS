
#include "main_file.h"

NNLS_solver::NNLS_solver(const Epetra_CrsMatrix &A, Epetra_MpiComm &Comm, Epetra_Vector &b, const int max_iter, const double tau):
        A_(A), Comm_(Comm), b_(b), x_(A.ColMap()), max_iter_(max_iter), tau_(tau), LS_iter_(1000), LS_tol_(10E-8){}

void NNLS_solver::Epetra_PermutationMatrix(std::vector<bool> &P, Epetra_CrsMatrix &P_mat){
  // Fill diagonal matrix with ones in the positive set
  // No longer in use
  double posOne = 1.0;
  for(int i = 0; i < P_mat.NumMyCols(); i++){
    int GlobalRow = P_mat.GRID(i);
    if (P[i] == 1) {
      P_mat.InsertGlobalValues(GlobalRow, 1, &posOne , &i);
    }
  }
}

void NNLS_solver::PositiveSetMatrix(std::vector<bool> &P,  Epetra_CrsMatrix &P_mat, Eigen::VectorXd &index_set){
  // Create matrix P_mat which contains the positive set of columns in A
  int colMap[A_.NumGlobalCols()];
  int numCol = 0;
  for(int j = 0; j < A_.NumGlobalCols(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int i =0; i < A_.NumGlobalRows(); i++){
    double row[A_.NumGlobalCols()];
    int numE;
    const int globalRow = A_.GRID(i);
    A_.ExtractGlobalRowCopy(globalRow, A_.NumGlobalCols(), numE , row);
    for(int j = 0; j < A_.NumGlobalCols(); j++){
      if (P[j] == 1) {
        P_mat.InsertGlobalValues(i, 1, &row[j] , &colMap[j]);
        
      }
    }
  }
}

void NNLS_solver::SubIntoX(Epetra_Vector &temp,  std::vector<bool> &P,  Eigen::VectorXd &index_set){
  // Substitute new values into the solution vector
  int colMap[x_.GlobalLength()];
  int numCol = 0;
  for(int j = 0; j < x_.GlobalLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x_.GlobalLength(); j++){
    if (P[j] == 1) {
      x_[j] = temp[colMap[j]];
    }
  }
}

void NNLS_solver::AddIntoX(Epetra_Vector &temp, std::vector<bool> &P, double alpha,  Eigen::VectorXd &index_set){
  // Add vector temp time scalar alpha into the vector x
  int colMap[x_.GlobalLength()];
  int numCol = 0;
  for(int j = 0; j < x_.GlobalLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x_.GlobalLength(); j++){
    if (P[j] == 1) {
      x_[j] += alpha*(temp[colMap[j]] -x_[j]);
    }
  }
}

void NNLS_solver::moveToActiveSet(int idx, Eigen::VectorXd &index_set, std::vector<bool> &P, std::vector<bool> &Z){
  // Move index at idx into the Active Set (Z set)
  P[index_set(idx)] = 0;
  Z[index_set(idx)] = 1; 

  std::swap(index_set(idx), index_set(numInactive_ - 1));
  numInactive_--;
}

void NNLS_solver::moveToInactiveSet(int idx, Eigen::VectorXd &index_set, std::vector<bool> &P, std::vector<bool> &Z){
  // Move index at idx into the Inactive Set (P set)
  P[index_set(idx)] = 1;
  Z[index_set(idx)] = 0;

  std::swap(index_set(idx), index_set(numInactive_));
  numInactive_++;
}

bool NNLS_solver::solve(){
  iter_ = 0;
  Eigen::VectorXd index_set;
  index_set = Eigen::VectorXd::LinSpaced(A_.NumMyCols(), 0, A_.NumMyCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
  numInactive_ = 0;
  std::vector<bool> Z(A_.NumMyCols());
  Z.flip();
  std::vector<bool> P(A_.NumMyCols());

  Epetra_CrsMatrix AtA(Epetra_DataAccess::View, A_.ColMap(), A_.NumMyCols());
  EpetraExt::MatrixMatrix::Multiply(A_, true, A_, false, AtA);

  Epetra_Vector Atb (A_.ColMap());
  A_.Multiply(true, b_, Atb);

  Epetra_Vector AtAx (A_.ColMap());
  Epetra_Vector Ax (A_.RowMap());
  Epetra_MultiVector gradient (A_.ColMap(), 1);
  Epetra_MultiVector residual (A_.RowMap(), 1);

  Eigen::VectorXd grad_eig (gradient.GlobalLength());
  Epetra_Vector grad_col (A_.ColMap());

  // OUTER LOOP
  while(true){
    // Early exit if all variables are inactive, which breaks 'maxCoeff' below.
    if (A_.NumGlobalCols() == numInactive_){
      return true;
    }

    AtA.Multiply(false, x_, AtAx);
    gradient = Atb;
    gradient.Update(-1.0, AtAx, 1.0);
    // std::cout << gradient << std::endl;

    grad_col = *gradient(0);
    for(int i = 0; i < gradient.GlobalLength() ; ++i){
      grad_eig[i] = grad_col[i];
    }
    
    // Find the maximum element of the gradient in the active set.
    // Move that variable to the inactive set.

    int numActive = A_.NumGlobalCols() - numInactive_;
    int argmaxGradient = -1;
    grad_eig(index_set.tail(numActive)).maxCoeff(&argmaxGradient);
    argmaxGradient += numInactive_;

    residual = b_;
    A_.Multiply(false, x_, Ax);
    residual.Update(-1.0, Ax, 1.0);
    // std::cout << residual << std::endl;
    double normRes[1];
    residual.Norm2(normRes);

    double normb[1];
    b_.Norm2(normb);
    // Exit Condition on the residual based on the norm of b
    if ((normRes[0]) <= (tau_ * normb[0])){
      return true;
    }
    
    moveToInactiveSet(argmaxGradient, index_set, P, Z);

    // INNER LOOP
    while(true){
      // Check if max. number of iterations is reached
      if (iter_ >= max_iter_){
        return false;
      }

      // Create matrix P_mat with columns from set P
      Epetra_Map Map(A_.NumGlobalRows(),0,Comm_);
      Epetra_Map ColMap(numInactive_,0,Comm_);
      Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, numInactive_);
      PositiveSetMatrix(P,  P_mat, index_set);
      P_mat.FillComplete(ColMap, Map);

      // Create temporary solution vector temp which is only the length of numInactive
      Epetra_Vector temp(P_mat.ColMap());

      // Solve least-squares problem in inactive set only with Aztec00
      Epetra_LinearProblem problem(&P_mat, &temp, &b_);
      AztecOO solver(problem);

      solver.SetAztecOption(AZ_conv, AZ_rhs);
      solver.SetAztecOption( AZ_precond, AZ_Jacobi);
      solver.SetAztecOption(AZ_output, AZ_none);
      solver.Iterate(LS_iter_, LS_tol_);
      iter_++; // The solve is expensive, so that is what we count as an iteration.
      
      // Check feasability...
      bool feasible = true;
      double alpha = Eigen::NumTraits<Eigen::VectorXd::Scalar>::highest();
      int infeasibleIdx = -1;
      for(int k = 0; k < numInactive_; k++){
        int idx = index_set[k];
        if (temp[k] < 0){
          // t should always be in [0,1]
          double t = -x_[idx]/(temp[k] - x_[idx]);
          if (alpha > t){
            alpha = t;
            infeasibleIdx = k;
            feasible = false;
          }
        }
      }
      eigen_assert(feasible || 0 <= infeasibleIdx);

      // If solution is feasible, exit to outer loop
      if (feasible){
        SubIntoX(temp, P, index_set);
        // std::cout << "sub temp: " << x << std::endl;
        break;
      }

      // Infeasible solution -> interpolate to feasible one
      AddIntoX(temp, P, alpha, index_set);
      // std::cout << "added with alpha: " << x << std::endl;

      // Remove these indices from the inactive set
      moveToActiveSet(infeasibleIdx, index_set, P, Z);
    }
    
  }
}