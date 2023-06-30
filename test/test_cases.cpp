#include "NNLS_solver.h"
#include <iostream>
#include <vector>
#include <fstream>

using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);

    // std::cout << indata.fail() << std::endl;

    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

bool verify_nnls_optimality(Eigen::MatrixXd &A, Eigen::MatrixXd &b_eig, Eigen::MatrixXd &x, double tau) {
  // The NNLS optimality conditions are:
  //
  // * 0 <= x[i] \forall i
  // * ||residual||_2 <= tau * ||b||_2 \forall i

  Eigen::MatrixXd res = (A * x - b_eig);
  bool opt = true;
  // NNLS solutions are EXACTLY not negative.
  if (0 > x.minCoeff()){
    opt = false;
  }
  else if (res.squaredNorm() > tau*b_eig.squaredNorm()){
    opt = false;
  }
  return opt;
}

void epetra_to_eig(int col, Epetra_Vector &x, Eigen::MatrixXd &x_eig){
  // Comvert epetra vector to eigen vector
  for(int i = 0; i < col; i++){
    x_eig(i,0) = x[i];
  }
}

bool test_nnls_known_CLASS(Eigen::MatrixXd &A_eig, int col, int row, Eigen::MatrixXd &x_eig, Eigen::MatrixXd &b_eig, double *b_pt, Epetra_MpiComm &Comm, const double tau, const int max_iter){
  // Check solution of NNLS problem with a known solution
  // Returns true if the solver exits for any condition other than max_iter and if the solution x accurate to the true solution and satisfies the conditions above
  Epetra_Map RowMap(row,0,Comm);
  Epetra_Map ColMap(col,0,Comm);
  Epetra_CrsMatrix A(Epetra_DataAccess::Copy, RowMap, col);
  const int numMyElements = RowMap.NumGlobalElements();

  for (int localRow = 0; localRow < numMyElements; ++localRow){
      const int globalRow = RowMap.GID(localRow);
      for(int n = 0 ; n < A_eig.cols() ; n++){
          A.InsertGlobalValues(globalRow, 1, &A_eig(globalRow, n), &n);
      }
  }

  A.FillComplete(ColMap, RowMap);
  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  Epetra_Vector b(Copy, A.RowMap(), b_pt);
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return exit_con;
  }

  Eigen::MatrixXd x_nnls_eig(col,1);
  epetra_to_eig(col, x , x_nnls_eig);
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  opt&= x_nnls_eig.isApprox(x_eig, tau);
  std::cout << opt << std::endl;
  return opt;
}

bool case_1 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4, 2);
  Eigen::MatrixXd x_eig(2,1);
  Eigen::MatrixXd b_eig(4,1);
  A_eig << 1, 1,  2, 4,  3, 9,  4, 16;
  b_eig << 0.6, 2.2, 4.8, 8.4;
  x_eig << 0.1, 0.5;
  double b_pt[] = {0.6, 2.2, 4.8, 8.4};

  return test_nnls_known_CLASS(A_eig, 2, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

bool case_2 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1,  1,  1,
       2,  4,  8,
       3,  9, 27,
       4, 16, 64;
  b_eig << 0.73, 3.24, 8.31, 16.72;
  x_eig << 0.1, 0.5, 0.13;
  double b_pt[] = {0.73, 3.24, 8.31, 16.72};

  return test_nnls_known_CLASS(A_eig, 3, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

bool case_3 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,4);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(4,1);

  A_eig << 1, 1, 1, 1, 2, 4, 8, 16, 3, 9, 27, 81, 4, 16, 64, 256;
  b_eig << 0.73, 3.24, 8.31, 16.72;
  x_eig << 0.1, 0.5, 0.13, 0;
  double b_pt[] = {0.73, 3.24, 8.31, 16.72};

  return test_nnls_known_CLASS(A_eig, 4, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

bool case_4 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b_eig << 0.23, 1.24, 3.81, 8.72;
  x_eig << 0.1, 0, 0.13;
  double b_pt[] = {0.23, 1.24, 3.81, 8.72};

  return test_nnls_known_CLASS(A_eig, 3, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

bool case_5 (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig(4,3);
  Eigen::MatrixXd b_eig(4,1);
  Eigen::MatrixXd x_eig(3,1);

  A_eig << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b_eig << 0.13, 0.84, 2.91, 7.12;
  x_eig << 0.0, 0.0, 0.1106544;
  double b_pt[] = {0.13, 0.84, 2.91, 7.12};

  return test_nnls_known_CLASS(A_eig, 3, 4, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}

bool case_MATLAB (Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  Eigen::MatrixXd A_eig = load_csv<MatrixXd>("C.csv");
  Eigen::MatrixXd b_eig = load_csv<MatrixXd>("d.csv");
  Eigen:: MatrixXd x_eig = load_csv<MatrixXd>("x_pow_4.csv");

  double *b_pt = b_eig.data();

  return test_nnls_known_CLASS(A_eig, 1024, 49, x_eig, b_eig, b_pt, Comm, tau, max_iter);
}


int main(int argc, char *argv[]){
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
  double tau = 1E-8;
  const int max_iter = 10000;

  bool ok = true;
  std::string p = "known";
  if (argv[1] == p){
    ok &= case_1(Comm, tau, max_iter);
    ok &= case_2(Comm, tau, max_iter);
    ok &= case_3(Comm, tau, max_iter);
    ok &= case_4(Comm, tau, max_iter);
    // ok &= case_5(Comm, tau, max_iter);
  }
  else{
    tau = 1E-4;
    ok &= case_MATLAB(Comm, tau, max_iter);
  }
  
  MPI_Finalize();

  if (ok) return 0;
  else return 1;
}