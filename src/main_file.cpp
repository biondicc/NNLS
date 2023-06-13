#include <iostream>
#include <string>
//#include "nnls_mod.h"
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
  double posOne = 1.0;
  for(int i = 0; i < P_mat.NumMyCols(); i++){
    int GlobalRow = P_mat.GRID(i);
    if (P[i] == 1) {
      P_mat.InsertGlobalValues(GlobalRow, 1, &posOne , &i);
    }
  }
}

void PositiveSetMatrix(std::vector<bool> &P,  Epetra_CrsMatrix &P_mat, const Epetra_CrsMatrix &A){
  int colMap[A.NumGlobalCols()];
  int numCol = 0;
  for(int j = 0; j < A.NumGlobalCols(); j++){
    if (P[j] == 1) {
      colMap[j] = numCol;
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

void SubIntoX(Epetra_Vector &temp, Epetra_Vector &x, std::vector<bool> &P){
  int colMap[x.GlobalLength()];
  int numCol = 0;
  for(int j = 0; j < x.GlobalLength(); j++){
    if (P[j] == 1) {
      colMap[j] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x.GlobalLength(); j++){
    if (P[j] == 1) {
      x[j] = temp[colMap[j]];
    }
  }
}

void AddIntoX(Epetra_Vector &temp, Epetra_Vector &x, std::vector<bool> &P, double alpha){
  int colMap[x.GlobalLength()];
  int numCol = 0;
  for(int j = 0; j < x.GlobalLength(); j++){
    if (P[j] == 1) {
      colMap[j] = numCol;
      numCol++;
    }
  }
  for(int j = 0; j < x.GlobalLength(); j++){
    if (P[j] == 1) {
      x[j] += alpha*(temp[colMap[j]] -x[j]);
    }
  }
}

void moveToActiveSet(int idx, int numInactive, Eigen::VectorXd index_set, std::vector<bool> &P, std::vector<bool> &Z){
  std::swap(index_set(idx), index_set(numInactive - 1));

  P[index_set(idx)] = 0;
  Z[index_set(idx)] = 1; 
}

void moveToInactiveSet(int idx, int numInactive, Eigen::VectorXd index_set, std::vector<bool> &P, std::vector<bool> &Z){
  std::swap(index_set(idx), index_set(numInactive));
  
  P[index_set(idx)] = 1;
  Z[index_set(idx)] = 0;
}

bool NNLS_solver(const Epetra_CrsMatrix &A, Epetra_MpiComm &Comm, Epetra_Vector &b, Epetra_Vector &x, const int max_iter, const double tau){
  int iter = 0;
  bool solve = true;
  Eigen::VectorXd index_set;
  index_set = Eigen::VectorXd::LinSpaced(A.NumMyCols(), 0, A.NumMyCols() -1);
  int numInactive = 0;
  std::vector<bool> Z(A.NumMyCols());
  Z.flip();
  std::vector<bool> P(A.NumMyCols());

  Epetra_CrsMatrix AtA(Epetra_DataAccess::Copy, A.ColMap(), A.NumMyCols());
  EpetraExt::MatrixMatrix::Multiply(A, true, A, false, AtA);
  //std::cout << AtA << std::endl;

  Epetra_Vector Atb (A.ColMap());
  A.Multiply(true, b, Atb);
  //std::cout << Atb << std::endl;

  Epetra_Vector AtAx (A.ColMap());
  Epetra_Vector Ax (A.RowMap());
  Epetra_MultiVector gradient (A.ColMap(), 1);
  Epetra_MultiVector residual (A.RowMap(), 1);

  Eigen::VectorXd grad_eig(gradient.GlobalLength());
  Epetra_Vector grad_col (A.ColMap());
  while(true){
    // std::cout << numInactive << std::endl;
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
    
    const int numActive = A.NumGlobalCols() - numInactive;
    int argmaxGradient = -1;
    grad_eig(index_set.tail(numActive)).maxCoeff(&argmaxGradient);
    argmaxGradient += numInactive;

    // ADD CHECK FOR RESIDUAL VALUE
    residual = b;
    A.Multiply(false, x, Ax);
    residual.Update(-1.0, Ax, 1.0);
    // std::cout << residual << std::endl;
    double normRes[1];
    residual.Norm2(normRes);

    double normb[1];
    b.Norm2(normb);
    if ((normRes[0]) <= (tau * normb[0])){
      return true;
    }
    
    moveToInactiveSet(argmaxGradient, numInactive, index_set, P, Z);
    numInactive++;
    Epetra_Map Map(A.NumGlobalRows(),0,Comm);
    Epetra_Map ColMap(numInactive,0,Comm);
    Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, numInactive);
    PositiveSetMatrix(P,  P_mat, A);
    P_mat.FillComplete(ColMap, Map);
    // std::cout << P_mat << std::endl;

    // Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, A.ColMap(), A.NumMyCols());
    // Epetra_PermutationMatrix(P, P_mat);
    // P_mat.FillComplete();
    // std::cout << P_mat << std::endl;

    // Epetra_CrsMatrix A_in_P(Epetra_DataAccess::Copy, A.RowMap(), A.NumMyCols());
    // EpetraExt::MatrixMatrix::Multiply(A, false, P_mat, false, A_in_P);
    // std::cout << A_in_P << std::endl;

    while(true){
      if (iter >= max_iter){
        return false;
      }

      Epetra_Vector temp(P_mat.ColMap());
      Epetra_LinearProblem problem(&P_mat, &temp, &b);
      // Create AztecOO instance
      AztecOO solver(problem);

      solver.SetAztecOption(AZ_conv, AZ_rhs);
      solver.SetAztecOption( AZ_precond, AZ_Jacobi);
      solver.SetAztecOption(AZ_output, AZ_none);
      solver.Iterate(100, 1.0E-5);

      // std::cout << temp << std::endl;
      iter++;
      bool feasible = true;
      double alpha = Eigen::NumTraits<Eigen::VectorXd::Scalar>::highest();
      int infeasibleIdx = -1;
      for(int k = 0; k < numInactive; k++){
        int idx = index_set[k];
        if (temp[idx] < 0){
          double t = -x[k]/(temp[k] - x[k]);
          if (alpha > t){
            alpha = t;
            infeasibleIdx = k;
            feasible = false;
          }
        }
      }
      eigen_assert(feasible || 0 <= infeasibleIdx);

      if (feasible){
        SubIntoX(temp, x, P);
        // std::cout << x << std::endl;
        break;
      }

      AddIntoX(temp, x, P, alpha);
      moveToActiveSet(infeasibleIdx, numInactive, index_set, P, Z);
      numInactive--;

      Epetra_Map Map(A.NumGlobalRows(),0,Comm);
      Epetra_Map ColMap(numInactive,0,Comm);
      Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, numInactive);
      PositiveSetMatrix(P,  P_mat, A);
      P_mat.FillComplete(ColMap, Map);
    }
    
  }
}

int main(int argc, char *argv[]) {
  using std::tuple;
  int status = 0;

  const int max_iter = 200;

  // Test 1
  Matrix<double, 4, 2> A_eig(4, 2);
  Matrix<double, 2, 1> x_eig(2);
  A_eig << 1, 1, 2, 4, 3, 9, 4, 16;
  double b_eig[] = {0.6, 2.2, 4.8, 8.4};
  int b_ind[] = {0, 1, 2, 3};
  x_eig << 0.1, 0.5;

  #ifdef HAVE_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
  #else
  Epetra_SerialComm Comm;
  #endif
  Epetra_Map Map(4,0,Comm);
  Epetra_Map ColMap(2,0,Comm);
  Epetra_CrsMatrix A(Epetra_DataAccess::Copy, Map, 2);
  const int numMyElements = Map.NumGlobalElements();

  for (int localRow = 0; localRow < numMyElements; ++localRow){
      const int globalRow = Map.GID(localRow);
      for(int n = 0 ; n < A_eig.cols() ; n++){
          A.InsertGlobalValues(globalRow, 1, &A_eig(globalRow, n), &n);
      }
  }

  A.FillComplete(ColMap, Map);
  std::cout << A << std::endl;
  
  Epetra_Vector x(A.ColMap());
  Epetra_Vector b(Copy, A.RowMap(), b_eig);
  std::cout << b;

  // Create Linear Problem
  Epetra_LinearProblem problem(&A, &x, &b);
  // Create AztecOO instance
  AztecOO solver(problem);

  solver.SetAztecOption(AZ_precond, AZ_Jacobi);
  solver.Iterate(100, 1.0E-8);

  std::cout << "Solver performed " << solver.NumIters() << " iterations." << std::endl
       << "Norm of true residual = " << solver.TrueResidual() << std::endl
       << x;
  
  Epetra_Vector x_new(A.ColMap());
  NNLS_solver(A, Comm, b, x_new, max_iter, 1.0E-8);
  std::cout << x_new << std::endl;

  // Test 2
  Matrix<double, 4, 3> A2_eig(4, 3);
  Matrix<double, 3, 1> x2_eig(3);
  A2_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  double b2_eig[] = {0.73, 3.24, 8.31, 16.72};
  x2_eig << 0.1, 0.5, 0.13;
  int b2_ind[] = {0, 1, 2, 3};

  Epetra_Map Map2(4,0,Comm);
  Epetra_Map ColMap2(3,0,Comm);
  Epetra_CrsMatrix A2(Epetra_DataAccess::Copy, Map2, 3);
  const int numMyElements2 = Map2.NumGlobalElements();

  for (int localRow = 0; localRow < numMyElements2; ++localRow){
      const int globalRow = Map2.GID(localRow);
      for(int n = 0 ; n < A2_eig.cols() ; n++){
          A2.InsertGlobalValues(globalRow, 1, &A2_eig(globalRow, n), &n);
      }
  }

  A2.FillComplete(ColMap2, Map2);
  std::cout << A2 << std::endl;
  
  Epetra_Vector x2(A2.ColMap());
  Epetra_Vector b2(Copy, A2.RowMap(), b2_eig);
  std::cout << b2;

  NNLS_solver(A2, Comm, b2, x2, max_iter, 1.0E-8);
  std::cout << x2;

  #ifdef HAVE_MPI
  MPI_Finalize();
  #endif
  return status;
}
