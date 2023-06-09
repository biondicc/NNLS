
#include <iostream>
#include <trilinos/Epetra_SerialComm.h>
#include <trilinos/Epetra_Map.h>
#include <trilinos/Epetra_CrsMatrix.h>
#include <trilinos/Epetra_Vector.h>

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
    Epetra_CrsMatrix _A;
    unsigned int _max_iter;
    float _epsilon;
public:
        NNLS_mod(const Epetra_CrsMatrix &A, unsigned int max_iter, float eps=1e-10);
        bool solve(const Epetra_Vector &b);
};

NNLS_mod::NNLS_mod(const Epetra_CrsMatrix &A, unsigned int max_iter, float eps=1e-10){
    Epetra_CrsMatrix _A(A);
    _max_iter = max_iter;
    _epsilon = eps;
}

bool NNLS_mod::solve(const Epetra_Vector &b){
    unsigned int _num_ls = 0;
    return true;
}