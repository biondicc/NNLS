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

void PositiveSetMatrix(std::vector<bool> &P,  Epetra_CrsMatrix &P_mat, const Epetra_CrsMatrix &A, Eigen::VectorXd &index_set){
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
  P[index_set(idx)] = 0;
  Z[index_set(idx)] = 1; 

  std::swap(index_set(idx), index_set(numInactive - 1));
}

void moveToInactiveSet(int idx, int numInactive, Eigen::VectorXd &index_set, std::vector<bool> &P, std::vector<bool> &Z){
  P[index_set(idx)] = 1;
  Z[index_set(idx)] = 0;

  std::swap(index_set(idx), index_set(numInactive));
}

bool NNLS_solver(const Epetra_CrsMatrix &A, Epetra_Comm &Comm, Epetra_Vector &b, Epetra_Vector &x, const int max_iter, const double tau){
  int iter = 0;
  bool solve = true;
  Eigen::VectorXd index_set;
  index_set = Eigen::VectorXd::LinSpaced(A.NumMyCols(), 0, A.NumMyCols() -1);
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
    // std::cout <<"index" << index_set << std::endl;
    numInactive++;
    std::cout << A.NumGlobalRows() << index_set << std::endl;
    
    std::cout << "map made";
    while(true){
      if (iter >= max_iter){
        return false;
      }
      Epetra_Map Map(A.NumGlobalRows(),0,Comm);
      Epetra_Map ColMap(numInactive,0,Comm);
      Epetra_CrsMatrix P_mat(Epetra_DataAccess::Copy, Map, numInactive);
      PositiveSetMatrix(P,  P_mat, A, index_set);
      P_mat.FillComplete(ColMap, Map);
      std::cout << P_mat << std::endl;
      Epetra_Vector temp(P_mat.ColMap());
      Epetra_LinearProblem problem(&P_mat, &temp, &b);
      // Create AztecOO instance
      AztecOO solver(problem);

      solver.SetAztecOption(AZ_conv, AZ_rhs);
      solver.SetAztecOption( AZ_precond, AZ_Jacobi);
      solver.SetAztecOption(AZ_output, AZ_none);
      solver.Iterate(1000, 1.0E-8);

      // std::cout << "temp: " << temp << std::endl;
      iter++;
      bool feasible = true;
      double alpha = Eigen::NumTraits<Eigen::VectorXd::Scalar>::highest();
      
      int infeasibleIdx = -1;
      for(int k = 0; k < numInactive; k++){
        int idx = index_set[k];
        if (temp[k] < 0){
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

      if (feasible){
        SubIntoX(temp, x, P, index_set);
        // std::cout << "sub temp: " << x << std::endl;
        break;
      }

      AddIntoX(temp, x, P, alpha, index_set);
      // std::cout << "added with alpha: " << x << std::endl;
      moveToActiveSet(infeasibleIdx, numInactive, index_set, P, Z);
      numInactive--;
    }
    
  }
}

// int main(int argc, char *argv[]) {
//   using std::tuple;
//   int status = 0;

//   #ifdef HAVE_MPI
//   MPI_Init(&argc,&argv);
//   Epetra_MpiComm Comm( MPI_COMM_WORLD );
//   #else
//   Epetra_SerialComm Comm;
//   #endif

//   const int max_iter = 3;

//   // Test 1
//   Eigen::MatrixXd A_eig(4, 2);
//   Eigen::MatrixXd x_eig(2,1);
//   A_eig << 1, 1, 2, 4, 3, 9, 4, 16;
//   double b_eig[] = {0.6, 2.2, 4.8, 8.4};
//   int b_ind[] = {0, 1, 2, 3};
//   x_eig << 0.1, 0.5;

//   std::cout << " Test 1 "<< std::endl;
//   test_nnls_known(A_eig, 2, 4, x_eig, b_eig, Comm);

//  /*  // Create Linear Problem
//   Epetra_LinearProblem problem(&A, &x, &b);
//   // Create AztecOO instance
//   AztecOO solver(problem);

//   solver.SetAztecOption(AZ_precond, AZ_Jacobi);
//   solver.Iterate(100, 1.0E-8);

//   std::cout << "Solver performed " << solver.NumIters() << " iterations." << std::endl
//        << "Norm of true residual = " << solver.TrueResidual() << std::endl
//        << x;
  
//   Epetra_Vector x_new(A.ColMap());
//   NNLS_solver(A, Comm, b, x_new, max_iter, 1.0E-8);
//   std::cout << x_new << std::endl;
//  */
//   // Test 2
//   Eigen::MatrixXd A2_eig(4, 3);
//   Eigen::MatrixXd x2_eig(3, 1);
//   A2_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
//   double b2_eig[] = {0.73, 3.24, 8.31, 16.72};
//   x2_eig << 0.1, 0.5, 0.13;

//   std::cout << " Test  "<< std::endl;
//   test_nnls_known(A2_eig, 3, 4, x2_eig, b2_eig, Comm);
  
//   // Test 3
//   Eigen::MatrixXd A3_eig(4, 4);
//   Eigen::MatrixXd x3_eig(4, 1);
//   A3_eig << 1, 1, 1, 1, 2, 4, 8, 16, 3, 9, 27, 81, 4, 16, 64, 256;
//   double b3_eig[] = {0.73, 3.24, 8.31, 16.72};
//   x3_eig << 0.1, 0.5, 0.13, 0;

//   std::cout << " Test 3 "<< std::endl;
//   test_nnls_known(A3_eig, 4, 4, x3_eig, b3_eig, Comm);
  
//   // Test 4
//   Eigen::MatrixXd A4_eig(4, 3);
//   Eigen::MatrixXd x4_eig(3, 1);
//   A4_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
//   double b4_eig[] = {0.23, 1.24, 3.81, 8.72};
//   x4_eig << 0.1, 0, 0.13;

//   std::cout << " Test 4 "<< std::endl;
//   test_nnls_known(A4_eig, 3, 4, x4_eig, b4_eig, Comm);
  

//   // Test 5
//   Eigen::MatrixXd A5_eig(4, 3);
//   Eigen::MatrixXd x5_eig(3, 1);
//   A5_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
//   double b5_eig[] = {0.13, 0.84, 2.91, 7.12};
//   // Solution obtained by original nnls() implementation in Fortran
//   x5_eig << 0.0, 0.0, 0.1106544;

//   std::cout << " Test 5 "<< std::endl;
//   test_nnls_known(A5_eig, 3, 4, x5_eig, b5_eig, Comm);

//   #ifdef HAVE_MPI
//   MPI_Finalize();
//   #endif
//   return status;
// }
