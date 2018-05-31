#include <Eigen/Dense>
#include <Eigen/Eigen>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


using namespace std;

int main()
{
    char Proj_dir[] = "/home/luk/PCA/Proj";
    char miu_dir[] = "/home/luk/PCA/mean";
    char eigen_dir[] = "/home/luk/PCA/eigen";
    ifstream in(Proj_dir);
    stringstream ss;
    double data[240];
    for (int i = 0; i < 30; ++i)
        for(int j = 0; j < 8; ++j)
        {
             in >> data[ i*8 + j];
        }
    in.close();
    Eigen::Map<const Eigen::Matrix<double, 30, 8>> proj_trans(data);
    Eigen::MatrixXd  m = proj_trans;
    const Eigen::Matrix<double, 8, 30> proj = proj_trans.transpose();
    cout << proj <<endl;

    in.open(eigen_dir);
    for(int j = 0; j < 30; ++j)
        {
             in >> data[j];
             data[j] = sqrt(data[j]);
             data[j] = 1.0 / data[j];
        }

    Eigen::Map<const Eigen::Matrix<double, 30, 1>> miu(data);
    cout << "miu" << endl; 
    cout << miu << endl; 

    const double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double b[3] = {7.0, 8.0, 9.0};
    Eigen::Map<const Eigen::Matrix<double, 3, 1> >  eigen_a(a);
    Eigen::Map<const Eigen::Matrix<double, 3, 1> >  eigen_b(b);

    Eigen::Matrix<double, 3, 1> product = eigen_a.array() * eigen_b.array();
    cout << product <<endl;

    Eigen::Matrix<double, 6, 1> M;

    M << eigen_a, eigen_b;
    std::cout << M << std::endl;

    Eigen::Matrix3d E;
    E = Eigen::MatrixXd::Identity(3, 3);
    
    cout << E << endl;

    E.inverse();

    cout << E << endl;

    Eigen::Quaterniond q;
    q = E;
    q.normalize();
    cout << q.x() << q.y() << q.z() << q.w() << endl;


    return 0;
}
