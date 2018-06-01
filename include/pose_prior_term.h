#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>


using namespace ceres;

class PoseCost_Project{
private:
    static const int m_row = 30;
    static const int m_col = 8;
    const Eigen::Matrix<double, m_row, m_col>& _PCA_proj;
    const Eigen::Matrix<double, m_row, 1>& _PCA_miu;

    const double pos_proj_weight;

public:
    PoseCost_Project(
            const Eigen::Matrix<double, m_row, m_col>& PCA_proj,
	          const Eigen::Matrix<double, m_row, 1>& PCA_miu,
            const double pos_proj_weight)
            :_PCA_proj(PCA_proj), _PCA_miu(PCA_miu), pos_proj_weight(pos_proj_weight){}

    template <typename T>
    bool operator()(const T* const spine_joint, const T* const lArm_joint, const T* const rArm_joint, T* error_pose_projection) const {
        Eigen::Matrix<T, m_row, 1> delta, residual;

        Eigen::Map<const Eigen::Matrix<T, 12, 1> > eigen_spine(spine_joint);
        Eigen::Map<const Eigen::Matrix<T, 9, 1> > eigen_rArm(rArm_joint);
        Eigen::Map<const Eigen::Matrix<T, 9, 1> > eigen_lArm(lArm_joint);

        Eigen::Matrix<T, m_row, 1> theta;
        theta << eigen_spine, eigen_rArm, eigen_lArm;


        delta = theta - _PCA_miu.cast<T>();

        Eigen::MatrixXd fullProj;
        fullProj = _PCA_proj * _PCA_proj.transpose();

        //residual = delta - _PCA_proj.cast<T>() * _PCA_proj.transpose().cast<T>() * delta;
        residual = delta - fullProj.cast<T>() * delta;
        error_pose_projection[0] = residual.transpose() * residual;
        error_pose_projection[0] *= (T)pos_proj_weight;

        return true;
   }
};

class PoseCost_Deviation{
private:
    static const int m_row = 30;
    static const int m_col = 8;
    const Eigen::Matrix<double, m_row, m_col>& _PCA_proj;
    const Eigen::Matrix<double, m_row, 1>& _PCA_miu;
    const Eigen::Matrix<double, m_col, 1>& _PCA_eigen;
    const double pos_dev_weight;

public:
    PoseCost_Deviation(
	     const Eigen::Matrix<double, m_row, m_col>& PCA_proj,
	     const Eigen::Matrix<double, m_row, 1>& PCA_miu,
	     const Eigen::Matrix<double, m_col, 1>& PCA_eigen,
       const double pos_dev_weight):
	     _PCA_proj(PCA_proj),
         _PCA_miu(PCA_miu),
         _PCA_eigen(PCA_eigen),
         pos_dev_weight(pos_dev_weight){}

    template <typename T>
    bool operator()(const T* const spine_joint, const T* const lArm_joint, const T* const rArm_joint, T* error_pose_deviation) const {
        Eigen::Matrix<T, m_row, 1> delta;
        Eigen::Matrix<T, m_col, 1> residual;
        Eigen::Map<const Eigen::Matrix<T, 12, 1> > eigen_spine(spine_joint);
        Eigen::Map<const Eigen::Matrix<T, 12, 1> > eigen_rArm(rArm_joint);
        Eigen::Map<const Eigen::Matrix<T, 12, 1> > eigen_lArm(lArm_joint);

        Eigen::Matrix<T, m_row, 1> theta;
        theta << eigen_spine, eigen_rArm, eigen_lArm;
        delta = theta - _PCA_miu.cast<T>();
	    residual = _PCA_eigen.cast<T>().array() * (_PCA_proj.transpose().cast<T>() * delta).array();

        error_pose_deviation[0] = residual.transpose() * residual;
        error_pose_deviation[0] *= (T)pos_dev_weight;

	return true;
    }
};
