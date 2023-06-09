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

bool NNLS_solver(const Epetra_CrsMatrix &A, const Epetra_Vector &b, Epetra_Vector &x, const int max_iter){
  int iter = 0;
  bool solve = true;
  Eigen::VectorXd index_set;
  index_set = Eigen::VectorXd::LinSpaced(A.NumMyCols(), 0, A.NumMyCols() -1);
  int numInactive = 0;
  
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
    if (A.NumGlobalCols() == numInactive){
      return true;
    }
    AtA.Multiply(false, x, AtAx);
    //std::cout << AtAx << std::endl;
    gradient = Atb;
    gradient.Update(-1.0, AtAx, 1.0);

    grad_col = *gradient(0);
    for(int i = 0; i < gradient.GlobalLength() ; ++i){
      grad_eig[i] = grad_col[i];
    }
    
    const int numActive = A.NumGlobalCols() - numInactive;
    int argmaxGradient = -1;
    grad_eig.maxCoeff(&argmaxGradient);
    argmaxGradient += numInactive;
    
    // ADD CHECK FOR RESIDUAL VALUE
    residual = b;
    A.Multiply(false, x, Ax);
    residual.Update(-1.0, Ax, 1.0);

    std::cout << residual << std::endl;
    return true;
  }
}

int main(int argc, char *argv[]) {
  using std::tuple;
  int status = 0;

  const int max_iter = 200;

  // Known Test Case
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
  
/* 
  int NumMyElements = 100;
  // Construct a Map that puts same number of equations on each processor
  Epetra_Map Map(-1, NumMyElements, 0, Comm);
  int NumGlobalElements = Map.NumGlobalElements();

  Epetra_CrsMatrix A(Copy, Map, 3);

  double negOne = -1.0;
  double posTwo = 2.0;
  for (int i=0; i<NumMyElements; i++) {
    int GlobalRow = A.GRID(i); int RowLess1 = GlobalRow - 1; int RowPlus1 = GlobalRow + 1;
    if (RowLess1!=-1) A.InsertGlobalValues(GlobalRow, 1, &negOne, &RowLess1);
    if (RowPlus1!=NumGlobalElements) A.InsertGlobalValues(GlobalRow, 1, &negOne, &RowPlus1);
    A.InsertGlobalValues(GlobalRow, 1, &posTwo, &GlobalRow);
    }

  A.FillComplete();
  std::cout << A;
 */
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
  NNLS_solver(A, b, x_new, max_iter);
  #ifdef HAVE_MPI
  MPI_Finalize();
  #endif
  return status;
}
