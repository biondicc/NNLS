
#include <iostream>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>

class NNLS_mod{
public:
    NNLS_mod(const Epetra_CrsMatrix &A, int max_iter = -1, float eps=1e-10):
    _max_iter(max_iter), _num_ls(0), _epsilon(eps), _A(A), _x(A.ColMap()){}

    bool solve(const Epetra_Vector &b);

protected:
    int _max_iter;
    int _num_ls;
    float _epsilon;
    Epetra_CrsMatrix _A;
    Epetra_Vector _x;
};

bool NNLS_mod::solve(const Epetra_Vector &b){

    _num_ls = 0;
}