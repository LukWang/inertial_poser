#pragma once
#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include <cmath>

template <typename T>
void EulerAnglesToRotationMatrixZXY(const T* euler, const int row_stride, T* R);

typedef struct keypoints_with_prob
{
  double x;
  double y;
  double p;
}KeyPoints;

template <typename T>
inline void EulerAnglesToRotationMatrixZXY(const T* euler, const int row_stride, T* R){
    const T degrees_to_radians(M_PI / 180.0);

    const T yaw(euler[0] * degrees_to_radians);
    const T pitch(euler[1] * degrees_to_radians);
    const T roll(euler[2] * degrees_to_radians);

    const T c1 = cos(yaw);
    const T s1 = sin(yaw);
    const T c2 = cos(pitch);
    const T s2 = sin(pitch);
    const T c3 = cos(roll);
    const T s3 = sin(roll);

  // Rows of the rotation matrix.
    T* R1 = R;
    T* R2 = R1 + row_stride;
    T* R3 = R2 + row_stride;

    R1[0] = c1*c3 - s1*s2*s3;
    R1[1] = -s1*c2;
    R1[2] = c1*s3+s1*s2*c3;

    R2[0] = s1*c3 + c1*s2*s3;
    R2[1] = c1*c2;
    R2[2] = s1*s3 - c1*s2*c3;

    R3[0] = -c2*s3;
    R3[1] = s2;
    R3[2] = c2*c3;
}

#endif
