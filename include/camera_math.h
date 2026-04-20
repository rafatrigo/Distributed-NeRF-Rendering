#pragma once
#include <Eigen/Dense>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// define a 4x4 Row-Major to ensure memory compatibility with numpy
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4fRM;

class CameraMath {
private:
    static Matrix4fRM trans_t(float t) {
        Matrix4fRM m = Matrix4fRM::Identity();
        m(2, 3) = t;
        return m;
    }

    static Matrix4fRM rot_phi(float phi) {
        Matrix4fRM m = Matrix4fRM::Identity();
        m(1, 1) = std::cos(phi);
        m(1, 2) = -std::sin(phi);
        m(2, 1) = std::sin(phi);
        m(2, 2) = std::cos(phi);
        return m;
    }

    static Matrix4fRM rot_theta(float th) {
        Matrix4fRM m = Matrix4fRM::Identity();
        m(0, 0) = std::cos(th);
        m(0, 2) = -std::sin(th);
        m(2, 0) = std::sin(th);
        m(2, 2) = std::cos(th);
        return m;
    }

public:
    static Matrix4fRM pose_spherical(float theta, float phi, float radius) {
        // degrees to radians
        float theta_rad = theta * M_PI / 180.0f;
        float phi_rad = phi * M_PI / 180.0f;

        // exact equivalent to the original python code
        Matrix4fRM c2w = trans_t(radius);
        c2w = rot_phi(phi_rad) * c2w;
        c2w = rot_theta(theta_rad) * c2w;

        // NeRF coordinate adjustment matrix ([-1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1])
        Matrix4fRM swap = Matrix4fRM::Zero();
        swap(0, 0) = -1.0f;
        swap(1, 2) =  1.0f;
        swap(2, 1) =  1.0f;
        swap(3, 3) =  1.0f;

        c2w = swap * c2w;
        
        return c2w;
    }
};
