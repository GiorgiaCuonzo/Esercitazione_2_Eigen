#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

VectorXd PALU_decomposition(const MatrixXd& A, const VectorXd& b, const int& num_system)
{
    float detA = A.determinant() ;
    if (detA == 0.0)
    {
        cerr << "The matrix A" << num_system << " is not invertible, therefore the LU decomposition with partial pivoting can not be used" << endl;
    }

    VectorXd x = A.lu().solve(b);

    return x;
}


VectorXd QR_decomposition(const MatrixXd& A, const VectorXd& b, const int& num_system)
{
    VectorXd x = A.householderQr().solve(b);

    //Or also it can be solved this way
    ColPivHouseholderQR<MatrixXd> qr(A);
    MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixR();
    VectorXd y = Q.transpose() * b;
    VectorXd xx = R.template triangularView<Upper>().solve(y);

    // They should give the same solution, if not there must be an error
    if (x != xx)
    {
        cerr << "The solution of the linear system num. " << num_system << " with QR decompostion given by the implemented command .householderQr().solve()"
                                                                      " does not coincide with the retrieved solution" << endl;
    }

    return xx;
}


double relative_error (const VectorXd& x_esatto, const VectorXd& x)
{
    double numeratore = (x_esatto-x).norm();
    double denominatore = x_esatto.norm();
    double err_rel = numeratore/denominatore;

    return err_rel;
}


main()
{
    Vector2d x_esatto = {-1.0e+0, -1.0e+00};

    //First system
    int num_system = 1;
    cout << "System num. " << num_system << endl;
    MatrixXd  A1(2,2);
    A1 << 5.547001962252291e-01,-3.770900990025203e-02, 8.320502943378437e-01,-9.992887623566787e-01;
    cout << "This is A" << num_system << ": " << endl << A1 << endl;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    cout << "This is b" << num_system << ": " << endl << b1 << endl;

    VectorXd x1_QR = QR_decomposition(A1,b1,num_system);
    cout << "This is the solution with QR decomposition: " << endl << setprecision(16) << x1_QR << endl;
    double err1_QR = relative_error(x_esatto, x1_QR);
    cout << "The respective relative error is: " << err1_QR << endl;

    VectorXd x1_LU = PALU_decomposition(A1,b1,num_system);
    cout << "This is the solution with PALU decomposition: " << endl << setprecision(16) << x1_LU << endl;
    double err1_LU = relative_error(x_esatto, x1_LU);
    cout << "The respective relative error is: " << err1_LU << endl << endl;


    //Second system
    num_system++;
    cout << "System num. " << num_system << endl;
    MatrixXd  A2(2,2);
    A2 << 5.547001962252291e-01,-5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    cout << "This is A" << num_system << ": " << endl << A2 << endl;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    cout << "This is b" << num_system << ": " << endl << b2 << endl;

    VectorXd x2_QR = QR_decomposition(A2,b2,num_system);
    cout << "This is the solution with QR decomposition: " << endl << setprecision(16) << x2_QR << endl;
    double err2_QR = relative_error(x_esatto, x2_QR);
    cout << "The respective relative error is: " << err2_QR << endl;

    VectorXd x2_LU = PALU_decomposition(A2,b2,num_system);
    cout << "This is the solution with PALU decomposition: " << endl << setprecision(16) << x2_LU << endl;
    double err2_LU = relative_error(x_esatto, x2_LU);
    cout << "The respective relative error is: " << err2_LU << endl << endl;


    //Third system
    num_system++;
    cout << "System num. " << num_system << endl;
    MatrixXd  A3(2,2);
    A3 << 5.547001962252291e-01,-5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
    cout << "This is A" << num_system << ": " << endl << A3 << endl;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    cout << "This is b" << num_system << ": " << endl << b3 << endl;

    VectorXd x3_QR = QR_decomposition(A3,b3,num_system);
    cout << "This is the solution with QR decomposition: " << endl << setprecision(16) << x3_QR << endl;
    double err3_QR = relative_error(x_esatto, x3_QR);
    cout << "The respective relative error is: " << err3_QR << endl;

    VectorXd x3_LU = PALU_decomposition(A3,b3,num_system);
    cout << "This is the solution with PALU decomposition: " << endl << setprecision(16) << x3_LU << endl;
    double err3_LU = relative_error(x_esatto, x3_LU);
    cout << "The respective relative error is: " << err3_LU << endl << endl;


    return 0;
}
