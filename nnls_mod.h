
#include <iostream>
#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include "eigen-3.4.0/Eigen/Dense"

// template <class Epetra_CrsMatrix> class NNLS_mod{
// public:
//     enum {
//         RowsAtCompileTime = Epetra_CrsMatrix::NumGlobalRows;
//         ColsAtCompileTime = Epetra_CrsMatrix::NumGlobalCols;
//     };

//     typedef Matrix<unsigned int, ColsAtCompileTime, 1> IndicesType;
    
//     Epetra_SerialComm Comm;
//     typedef Epetra_Map<ColsAtCompileTime,0,Comm> SolutionMap;
//     typedef Epetra_CrsMatrix<Epetra_DataAccess::Copy,SolutionMap, 1> SolutionVectorType;

//     NNLS_mod();

//     NNLS_mod(const Epetra_CrsMatrix &A, int max_iter = -1, float eps=1e-10);

//     template <typename Epetra_CrsMatrix> NNLS_mod<Epetra_CrsMatrix> &compute(const Epetra_CrsMatrix &A);


//     bool solve(const Epetra_Vector &b);

// protected:
//     unsigned int _max_iter;
//     unsigned int _num_ls;
//     float _epsilon;
//     Epetra_CrsMatrix _A;
//     Epetra_Vector _x;
//     unsigned int _numInactive;
//     const int _numCols;
    
// };

class NNLS_mod{
    const Epetra_CrsMatrix A; 
    unsigned int max_iter;
    float eps;
    public:
        NNLS_mod (const Epetra_CrsMatrix, unsigned int, float);
        bool solve(const Epetra_Vector &b);
};

bool NNLS_mod::solve(const Epetra_Vector &b){
    unsigned int _num_ls = 0;
    return true;
}