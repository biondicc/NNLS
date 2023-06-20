#include <iostream>
#include <string>
#include <AztecOO_config.h>
#ifdef HAVE_MPI
#include <mpi.h>
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif
#include <Epetra_ConfigDefs.h>


#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_LinearProblem.h>
#include <EpetraExt_MatrixMatrix.h>
#include <AztecOO.h>
#include <Eigen/Dense>

using Eigen::Matrix;

void Epetra_PermutationMatrix(std::vector<bool> &P, Epetra_CrsMatrix &P_mat){
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

void PositiveSetMatrix(std::vector<bool> &P,  Epetra_CrsMatrix &P_mat, const Epetra_CrsMatrix &A, Eigen::VectorXd &index_set){
  // Create matrix P_mat which contains the positive set of columns in A
  int colMap[A.NumGlobalCols()];
  int numCol = 0;
  for(int j = 0; j < A.NumGlobalCols(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int i =0; i < A.NumGlobalRows(); i++){
    double row[A.NumGlobalCols()];
    int numE;
    const int globalRow = A.GRID(i);
    A.ExtractGlobalRowCopy(globalRow, A.NumGlobalCols(), numE , row);
    for(int j = 0; j < A.NumGlobalCols(); j++){
      if (P[j] == 1) {
        P_mat.InsertGlobalValues(i, 1, &row[j] , &colMap[j]);
        
      }
    }
  }
}

void SubIntoX(Epetra_Vector &temp, Epetra_Vector &x, std::vector<bool> &P,  Eigen::VectorXd &index_set){
  // Substitute new values into the solution vector
  int colMap[x.GlobalLength()];
  int numCol = 0;
  for(int j = 0; j < x.GlobalLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x.GlobalLength(); j++){
    if (P[j] == 1) {
      x[j] = temp[colMap[j]];
    }
  }
}

void AddIntoX(Epetra_Vector &temp, Epetra_Vector &x, std::vector<bool> &P, double alpha,  Eigen::VectorXd &index_set){
  // Add vector temp time scalar alpha into the vector x
  int colMap[x.GlobalLength()];
  int numCol = 0;
  for(int j = 0; j < x.GlobalLength(); j++){
    int idx = index_set[j];
    if (P[idx] == 1) {
      colMap[idx] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x.GlobalLength(); j++){
    if (P[j] == 1) {
      x[j] += alpha*(temp[colMap[j]] -x[j]);
    }
  }
}

void moveToActiveSet(int idx, int numInactive, Eigen::VectorXd &index_set, std::vector<bool> &P, std::vector<bool> &Z){
  // Move index at idx into the Active Set (Z set)
  P[index_set(idx)] = 0;
  Z[index_set(idx)] = 1; 

  std::swap(index_set(idx), index_set(numInactive - 1));
}

void moveToInactiveSet(int idx, int numInactive, Eigen::VectorXd &index_set, std::vector<bool> &P, std::vector<bool> &Z){
  // Move index at idx into the Inactive Set (P set)
  P[index_set(idx)] = 1;
  Z[index_set(idx)] = 0;

  std::swap(index_set(idx), index_set(numInactive));
}

bool NNLS_solver(const Epetra_CrsMatrix &A, Epetra_Comm &Comm, Epetra_Vector &b, Epetra_Vector &x, const int max_iter, const double tau, int LS_iter = 1000, double LS_tol = 10E-8){
  int iter = 0;
  bool solve = true;
  Eigen::VectorXd index_set;
  index_set = Eigen::VectorXd::LinSpaced(A.NumMyCols(), 0, A.NumMyCols() -1); // Indices proceeding and including numInactive are in the P set (Inactive/Positive)
  int numInactive = 0;
  std::vector<bool> Z(A.NumMyCols());
  Z.flip();
  std::vector<bool> P(A.NumMyCols());

  Epetra_CrsMatrix AtA(Epetra_DataAccess::View, A.ColMap(), A.NumMyCols());
  // std::cout << AtA << std::endl;
  EpetraExt::MatrixMatrix::Multiply(A, true, A, false, AtA);

  Epetra_Vector Atb (A.ColMap());
  A.Multiply(true, b, Atb);
  // std::cout << Atb << std::endl;

  Epetra_Vector AtAx (A.ColMap());
  Epetra_Vector Ax (A.RowMap());
  Epetra_MultiVector gradient (A.ColMap(), 1);
  Epetra_MultiVector residual (A.RowMap(), 1);

  Eigen::VectorXd grad_eig(gradient.GlobalLength());
  Epetra_Vector grad_col (A.ColMap());

  // OUTER LOOP
  while(true){
    // Early exit if all variables are inactive, which breaks 'maxCoeff' below.
    if (A.NumGlobalCols() == numInactive){
      return true;
    }

    AtA.Multiply(false, x, AtAx);
    // std::cout << AtAx << std::endl;
    gradient = Atb;
    gradient.Update(-1.0, AtAx, 1.0);
    // std::cout << gradient << std::endl;

    grad_col = *gradient(0);
    for(int i = 0; i < gradient.GlobalLength() ; ++i){
      grad_eig[i] = grad_col[i];
    }
    
    // Find the maximum element of the gradient in the active set.
    // Move that variable to the inactive set.

    const int numActive = A.NumGlobalCols() - numInactive;
    int argmaxGradient = -1;
    grad_eig(index_set.tail(numActive)).maxCoeff(&argmaxGradient);
    argmaxGradient += numInactive;

    residual = b;
    A.Multiply(false, x, Ax);
    residual.Update(-1.0, Ax, 1.0);
    // std::cout << residual << std::endl;
    double normRes[1];
    residual.Norm2(normRes);

    double normb[1];
    b.Norm2(normb);
    // Exit Condition on the residual based on the norm of b
    if ((normRes[0]) <= (tau * normb[0])){
      return true;
    }
    
    moveToInactiveSet(argmaxGradient, numInactive, index_set, P, Z);
    // std::cout <<"index" << index_set << std::endl;
    numInactive++;

    // INNER LOOP
    while(true){
      // Check if max. number of iterations is reached
      if (iter >= max_iter){
        return false;
      }

      // Create matrix P_mat with columns from set P
      Epetra_Map Map(A.NumGlobalRows(),0,Comm);
      Epetra_Map ColMap(numInactive,0,Comm);
      Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, numInactive);
      PositiveSetMatrix(P,  P_mat, A, index_set);
      P_mat.FillComplete(ColMap, Map);
      // std::cout << P_mat << std::endl;

      // Create temporary solution vector temp which is only the length of numInactive
      Epetra_Vector temp(P_mat.ColMap());

      // Solve least-squares problem in inactive set only with Aztec00
      Epetra_LinearProblem problem(&P_mat, &temp, &b);
      AztecOO solver(problem);

      solver.SetAztecOption(AZ_conv, AZ_rhs);
      solver.SetAztecOption( AZ_precond, AZ_Jacobi);
      solver.SetAztecOption(AZ_output, AZ_none);
      solver.Iterate(LS_iter, LS_tol);
      // std::cout << "temp: " << temp << std::endl;
      iter++; // The solve is expensive, so that is what we count as an iteration.
      
      // Check feasability...
      bool feasible = true;
      double alpha = Eigen::NumTraits<Eigen::VectorXd::Scalar>::highest();
      int infeasibleIdx = -1;
      for(int k = 0; k < numInactive; k++){
        int idx = index_set[k];
        if (temp[k] < 0){
          // t should always be in [0,1]
          // std::cout << "temp[k]: " << temp[k] << std::endl;
          double t = -x[idx]/(temp[k] - x[idx]);
          // std::cout << "t: " << t << std::endl; 
          if (alpha > t){
            alpha = t;
            // std::cout << "alpha: " << alpha << std::endl;
            infeasibleIdx = k;
            feasible = false;
          }
        }
      }
      eigen_assert(feasible || 0 <= infeasibleIdx);

      // If solution is feasible, exit to outer loo
      if (feasible){
        SubIntoX(temp, x, P, index_set);
        // std::cout << "sub temp: " << x << std::endl;
        break;
      }

      // Infeasible solution -> interpolate to feasible one
      AddIntoX(temp, x, P, alpha, index_set);
      // std::cout << "added with alpha: " << x << std::endl;

      // Remove these indices from the inactive set
      moveToActiveSet(infeasibleIdx, numInactive, index_set, P, Z);
      numInactive--;
    }
    
  }
}