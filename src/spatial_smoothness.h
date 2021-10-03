#include <vector>

#include "ceres/cost_function.h"


class SmoothnessCostFunction: public ceres::CostFunction
{
public:
    explicit SmoothnessCostFunction(std::vector<double>& weights_): weights(weights_)
    {
        set_num_residuals(1);

        for (int i = 0; i < weights_.size(); i++)
        {
            mutable_parameter_block_sizes()->push_back(1);
        }
    }

    virtual bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override
    {
        residuals[0] = 0.0;
        int num_variables = weights.size();

        for (int i = 1; i < num_variables; i++)
        {
            residuals[0] += weights[i] * (parameters[0][0] - parameters[i][0]);
        }

        if (jacobians != NULL)
        {
            jacobians[0][0] = weights[0];
            for (int i = 1; i < num_variables; i++)
            {
                if (jacobians[i] != NULL) {
                    jacobians[i][0] = -weights[i];
                }
            }
        }

        return true;
    }

private:
    std::vector<double> weights;
};
