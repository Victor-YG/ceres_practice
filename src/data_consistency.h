#include "ceres/cost_function.h"


class DataConsistencyCostFunction: public ceres::SizedCostFunction<1, 1>
{
public:
    explicit DataConsistencyCostFunction(double d_): d(d_) {};

    virtual bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override 
    {
        residuals[0] = parameters[0][0] - d;
        if (jacobians != NULL && jacobians[0] != NULL)
        {
            jacobians[0][0] = 1;
        }
        return true;
    }

private:
    double d;
};
