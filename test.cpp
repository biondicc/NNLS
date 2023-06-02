
#include <iostream>
#include <Epetra_SerialComm.h>
#include "nnls_mod.h"

int main(){
    Epetra_SerialComm Comm;

    // Epetra_Map Map(8,0,Comm);
    // Epetra_CrsMatrix A(Epetra_DataAccess::Copy, Map, 2);
    // NNLS_mod nnls (A, 100, 1e-10);

    return 0;
}