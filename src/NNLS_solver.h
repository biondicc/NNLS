#include <iostream>
#include <string>
#include <AztecOO_config.h>
#include <mpi.h>
#include <Epetra_MpiComm.h>
#include <Epetra_ConfigDefs.h>


#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_LinearProblem.h>
#include <EpetraExt_MatrixMatrix.h>
#include <AztecOO.h>
#include <Amesos.h>
#include <Amesos_BaseSolver.h>
#include <Eigen/Dense>

using Eigen::Matrix;

class NNLS_solver
{
public:
    /// Default constructor that will set the constants.
    NNLS_solver(const Epetra_CrsMatrix &A, Epetra_MpiComm &Comm, Epetra_Vector &b, const int max_iter, const double tau); ///< Constructor.
    virtual ~NNLS_solver() {}; ///< Destructor.
    bool solve();
    Epetra_Vector & getSolution() {return x_;}
  
protected:
    const Epetra_CrsMatrix A_;
    Epetra_MpiComm Comm_;
    Epetra_Vector b_;
    Epetra_Vector x_;
    const int max_iter_;
    const double tau_;
    int LS_iter_;
    double LS_tol_;
    Eigen::VectorXd index_set;
    std::vector<bool> Z;
    std::vector<bool> P;

public:
    int iter_;
    int numInactive_;

private:
    void Epetra_PermutationMatrix(Epetra_CrsMatrix &P_mat);
    void PositiveSetMatrix(Epetra_CrsMatrix &P_mat);
    void SubIntoX(Epetra_Vector &temp);
    void AddIntoX(Epetra_Vector &temp, double alpha);
    void moveToActiveSet(int idx);
    void moveToInactiveSet(int idx);

};
