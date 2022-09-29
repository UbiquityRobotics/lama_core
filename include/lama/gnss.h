
#pragma once

#include "utm.h"

namespace lama {

// Global Navigation Satellite System
struct GNSS {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // GNSS fix status, -1 is no fix
    int16_t status = -1;

    double latitude;
    double longitude;

    Matrix2d covar;

    inline void toUTM(double& x, double& y, std::string& zone) const
    { LLtoUTM(latitude, longitude, y, x, zone); }

    inline void fromUTM(double x, double y, const std::string& zone)
    { UTMtoLL(y, x, zone, latitude, longitude); }
};

}// namespace lama
