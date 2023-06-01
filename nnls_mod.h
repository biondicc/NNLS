
#include <iostream>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <eigen/Eigen/Dense>

class NNLS_mod{
public:
    NNLS_mod(const Epetra_CrsMatrix &A, int max_iter = -1, float eps=1e-10):
    _max_iter(max_iter), _num_ls(0), _epsilon(eps), _A(A), _x(A.ColMap()), _numInactive(0){}

    bool solve(const Epetra_Vector &b);
    typedef Eigen::Matrix<int, A.NumGlobalCols(), 1> IndicesType;

protected:
    unsigned int _max_iter;
    unsigned int _num_ls;
    float _epsilon;
    Epetra_CrsMatrix _A;
    Epetra_Vector _x;
    unsigned int _numInactive;
    IndicesType _index_sets;
};

bool NNLS_mod::solve(const Epetra_Vector &b){
    _num_ls = 0;
    _index_sets = IndicesType::LinSpaced(_A.NumGlobalCols(), 0, _A.NumGlobalCols()- 1);
    _numInactive = 0;

}