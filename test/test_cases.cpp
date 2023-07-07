#include "NNLS_solver.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/QR>
#include <test/random_matrix_helper.h>

using namespace Eigen;
#define EIGEN_TEST_MAX_SIZE 50

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

bool verify_nnls_optimality(Eigen::MatrixXd &A, Eigen::MatrixXd &b_eig, Eigen::MatrixXd &x, double tau, bool random = false) {
  // The NNLS optimality conditions are:
  //
  // * 0 <= x[i] \forall i
  // * ||residual||_2 <= tau * ||b||_2 \forall i

  Eigen::MatrixXd res = (b_eig - A * x);
  bool opt = true;
  // NNLS solutions are EXACTLY not negative.
  if (0 > x.minCoeff()){
    opt = false;
  }
  else if (res.squaredNorm() > tau*b_eig.squaredNorm()){
    opt = false;
  }
  if (random){
    if ((A.transpose() * res).maxCoeff() < tau){
      opt = true;
      std::cout << "Gradient is small/negative, but residual is not necessarily small" << std::endl;
    }
  }
  return opt;
}

void epetra_to_eig_vec(int col, Epetra_Vector &x, Eigen::MatrixXd &x_eig){
  // Convert epetra vector to eigen vector
  for(int i = 0; i < col; i++){
    x_eig(i,0) = x[i];
  }
}

Epetra_CrsMatrix eig_to_epetra_matrix(Eigen::MatrixXd &A_eig, int col, int row, Epetra_MpiComm &Comm){
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
  return A;
}

bool test_nnls_known_CLASS(Eigen::MatrixXd &A_eig, int col, int row, Eigen::MatrixXd &x_eig, Eigen::MatrixXd &b_eig, double *b_pt, Epetra_MpiComm &Comm, const double tau, const int max_iter){
  // Check solution of NNLS problem with a known solution
  // Returns true if the solver exits for any condition other than max_iter and if the solution x accurate to the true solution and satisfies the conditions above
  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, col, row, Comm));
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
  epetra_to_eig_vec(col, x , x_nnls_eig);
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  opt&= x_nnls_eig.isApprox(x_eig, tau);
  std::cout << opt << std::endl;
  return opt;
}

bool test_nnls_handles_Mx0_matrix(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  //
  // SETUP
  //
  const int row = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig(row, 0);
  VectorXd b_eig = VectorXd::Random(row);

  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, 0, row, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  //
  // ACT
  //
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return exit_con;
  }

  bool opt = true;
  opt &= (NNLS_prob.iter_ == 0);
  Epetra_Vector x(NNLS_prob.getSolution());
  opt &= (x.MyLength()== 0);
  return opt;
}

bool test_nnls_handles_0x0_matrix(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  //
  // SETUP
  //
  MatrixXd A_eig(0, 0);
  VectorXd b_eig(0);

  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, 0, 0, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  //
  // ACT
  //
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return exit_con;
  }

  bool opt = true;
  opt &= (NNLS_prob.iter_ == 0);
  Epetra_Vector x(NNLS_prob.getSolution());
  opt &= (x.MyLength()== 0);
  return opt;
}

bool test_nnls_random_problem(Epetra_MpiComm &Comm) {
  //
  // SETUP
  //

  const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  // Make some sort of random test problem from a wide range of scales and condition numbers.
  using std::pow;
  using Scalar = typename MatrixXd::Scalar;
  const Scalar sqrtConditionNumber = pow(Scalar(10), internal::random<Scalar>(Scalar(0), Scalar(2)));
  const Scalar scaleA = pow(Scalar(10), internal::random<Scalar>(Scalar(-3), Scalar(3)));
  const Scalar minSingularValue = scaleA / sqrtConditionNumber;
  const Scalar maxSingularValue = scaleA * sqrtConditionNumber;
  MatrixXd A_eig(rows, cols);
  generateRandomMatrixSvs(setupRangeSvs<Matrix<Scalar, Dynamic, 1>>(cols, minSingularValue, maxSingularValue), rows,
                          cols, A_eig);

  // Make a random RHS also with a random scaling.
  using VectorB = decltype(A_eig.col(0).eval());
  MatrixXd b_eig = 100 * VectorB::Random(A_eig.rows());

  //
  // ACT
  //

  using Scalar = typename MatrixXd::Scalar;
  using std::sqrt;
   const Scalar tolerance =
       sqrt(Eigen::GenericNumTraits<Scalar>::epsilon()) * b_eig.cwiseAbs().maxCoeff() * A_eig.cwiseAbs().maxCoeff();
  Index max_iter = 100 * A_eig.cols();  // A heuristic guess.
  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tolerance);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //

  // In fact, NNLS can fail on some problems, but they are rare in practice.
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }

  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tolerance, true);
  return opt;
}

bool test_nnls_handles_zero_rhs(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  //
  // SETUP
  //
  const Index cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig = MatrixXd::Random(rows, cols);
  MatrixXd b_eig = VectorXd::Zero(rows);

  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;
  //
  // ACT
  //
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    return false;
  }

  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  bool opt = true;
  opt &= (NNLS_prob.iter_ <= 1);
  opt &= (x_nnls_eig.isApprox(VectorXd::Zero(cols)));
  return opt;
}

bool test_nnls_handles_dependent_columns(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  //
  // SETUP
  //
  const Index rank = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE / 2);
  const Index cols = 2 * rank;
  const Index rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  MatrixXd A_eig = MatrixXd::Random(rows, rank) * MatrixXd::Random(rank, cols);
  MatrixXd b_eig = VectorXd::Random(rows);
  
  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;
  //
  // ACT
  //
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //
  // What should happen when the input 'A' has dependent columns?
  // We might still succeed. Or we might not converge.
  // Either outcome is fine. If Success is indicated,
  // then 'x' must actually be a solution vector.
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
  }

  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;

  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  bool opt = verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau, true);
  return opt;
}

bool test_nnls_handles_wide_matrix(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  //
  // SETUP
  //
  const Index cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
  const Index rows = internal::random<Index>(2, cols - 1);
  MatrixXd A_eig = MatrixXd::Random(rows, cols);
  MatrixXd b_eig = VectorXd::Random(rows);
  
  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, cols, rows, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;
  //
  // ACT
  //
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //
  // What should happen when the input 'A' is wide?
  // The unconstrained least-squares problem has infinitely many solutions.
  // Subject the the non-negativity constraints,
  // the solution might actually be unique (e.g. it is [0,0,..,0]).
  // So, NNLS might succeed or it might fail.
  // Either outcome is fine. If Success is indicated,
  // then 'x' must actually be a solution vector.


  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  bool opt = true;
  Eigen::MatrixXd x_nnls_eig(cols,1);
  epetra_to_eig_vec(cols, x , x_nnls_eig);
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau, true);
  }
  else{
    opt &= verify_nnls_optimality(A_eig, b_eig, x_nnls_eig, tau);
  }
  return opt;
}

bool test_nnls_special_case_solves_in_zero_iterations(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness.
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive.
  MatrixXd b_eig = A_eig * x_eig;


  double *b_pt = b_eig.data();
  double *x_pt = x_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);
  Epetra_Vector x_start(Copy, A.ColMap(), x_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  //
  // ACT
  //
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  NNLS_prob.startingSolution(x_start);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //

  bool opt = true;
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  opt &= (NNLS_prob.iter_ == 0);
  return opt;
}

bool test_nnls_special_case_solves_in_n_iterations(Epetra_MpiComm &Comm, const double tau, const int max_iter) {
  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness.
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive.
  MatrixXd b_eig = A_eig * x_eig;

  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  //
  // ACT
  //
  NNLS_solver NNLS_prob(A, Comm, b, max_iter, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //

  bool opt = true;
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt &= false;
  }
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  opt &= (NNLS_prob.iter_ == n);
  return opt;
}

bool test_nnls_returns_NoConvergence_when_maxIterations_is_too_low(Epetra_MpiComm &Comm, const double tau) {
  // Using the special case that takes `n` iterations,
  // from `test_nnls_special_case_solves_in_n_iterations`,
  // we can set max iterations too low and that should cause the solve to fail.

  const Index n = 10;
  const Index m = 3 * n;
  // With high probability, this is full column rank, which we need for uniqueness.
  MatrixXd A_eig = MatrixXd::Random(m, n);
  MatrixXd x_eig = VectorXd::Random(n).cwiseAbs().array() + 1;  // all positive.
  MatrixXd b_eig = A_eig * x_eig;

  double *b_pt = b_eig.data();

  Epetra_CrsMatrix A(eig_to_epetra_matrix(A_eig, n, m, Comm));
  Epetra_Vector b(Copy, A.RowMap(), b_pt);

  std::cout << " Matrix A "<< std::endl;
  std::cout << A << std::endl;
  
  std::cout << " Vector b "<< std::endl;
  std::cout << b << std::endl;

  //
  // ACT
  //
  const Index max_iters = n - 1;
  NNLS_solver NNLS_prob(A, Comm, b, max_iters, tau);
  bool exit_con = NNLS_prob.solve();

  //
  // VERIFY
  //
  bool opt = false;
  if (!exit_con){
    std::cout << "Exited due to maximum iterations, not necessarily optimum solution" << std::endl;
    opt = true;
  }
  std::cout << " Solution x "<< std::endl;
  Epetra_Vector x(NNLS_prob.getSolution());
  std::cout << x << std::endl;
  opt &= (NNLS_prob.iter_ == max_iters);
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

  std::string s_1 = "known";
  std::string s_2 = "matlab";
  std::string s_3 = "noCols";
  std::string s_4 = "empty";
  std::string s_5 = "random";
  std::string s_6 = "zeroRHS";
  std::string s_7 = "depCols";
  std::string s_8 = "wide";
  std::string s_9 = "zeroIter";
  std::string s_10 = "nIter"; 
  std::string s_11 = "maxIter"; 
  if (argv[1] == s_1){
    ok &= case_1(Comm, tau, max_iter);
    ok &= case_2(Comm, tau, max_iter);
    ok &= case_3(Comm, tau, max_iter);
    ok &= case_4(Comm, tau, max_iter);
    // ok &= case_5(Comm, tau, max_iter);
  }
  else if (argv[1] == s_2){
    tau = 1E-4;
    ok &= case_MATLAB(Comm, tau, max_iter);
  }
  else if (argv[1] == s_3){
    ok &= test_nnls_handles_Mx0_matrix(Comm, tau, max_iter);
  }  
  else if (argv[1] == s_4){
    ok &= test_nnls_handles_0x0_matrix(Comm, tau, max_iter);
  }
  else if (argv[1] == s_5){
    ok &= test_nnls_random_problem(Comm);
  }
  else if (argv[1] == s_6){
    ok &= test_nnls_handles_zero_rhs(Comm, tau, max_iter);
  }
  else if (argv[1] == s_7){
    ok &= test_nnls_handles_dependent_columns(Comm, tau, max_iter);
  }
  else if (argv[1] == s_8){
    ok &= test_nnls_handles_wide_matrix(Comm, tau, max_iter);
  }
  else if (argv[1] == s_9){
    ok &= test_nnls_special_case_solves_in_zero_iterations(Comm, tau, max_iter);
  }
  else if (argv[1] == s_10){
    ok &= test_nnls_special_case_solves_in_n_iterations(Comm, tau, max_iter);
  }
  else if (argv[1] == s_11){
    ok &= test_nnls_returns_NoConvergence_when_maxIterations_is_too_low(Comm, tau);
  }
  else {
    ok = false;
  }
  MPI_Finalize();

  if (ok) return 0;
  else return 1;
}