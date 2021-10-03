#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "opencv4/opencv2/imgcodecs.hpp"

#include "utils.h"
#include "data_consistency.h"
#include "spatial_smoothness.h"


// input variables
DEFINE_string(input, "", "File to be denoised");
DEFINE_string(output, "", "File to which the output image should be written");
DEFINE_double(lambda, 1.0, "Weight on spatial smoothness term. Default: 1.0");
DEFINE_int32(x_min, -1, "x_min of roi");
DEFINE_int32(x_max, -1, "x_max of roi");
DEFINE_int32(y_min, -1, "y_min of roi");
DEFINE_int32(y_max, -1, "y_max of roi");

DEFINE_double(sigma, 20.0, "Standard deviation of noise");
DEFINE_string(trust_region_strategy, "levenberg_marquardt", "Options are: levenberg_marquardt, dogleg.");
DEFINE_string(dogleg, "traditional_dogleg", "Options are: traditional_dogleg, subspace_dogleg.");
DEFINE_string(linear_solver, "sparse_normal_cholesky", "Options are: " "sparse_normal_cholesky and cgnr.");
DEFINE_string(preconditioner, "jacobi", "Options are: " "identity, jacobi, subset");
DEFINE_string(sparse_linear_algebra_library, "suite_sparse", "Options are: suite_sparse, cx_sparse and eigen_sparse");
DEFINE_double(eta, 1e-2, "Default value for eta. Eta determines the accuracy of each linear solve of the truncated newton step."
                         "Changing this parameter can affect solve performance.");
DEFINE_int32(num_threads, 1, "Number of threads.");
DEFINE_int32(num_iterations, 10, "Number of iterations.");
DEFINE_bool(nonmonotonic_steps, false, "Trust region algorithm can use" " nonmonotic steps.");
DEFINE_bool(inner_iterations, false, "Use inner iterations to non-linearly refine each successful trust region step.");
DEFINE_bool(mixed_precision_solves, false, "Use mixed precision solves.");
DEFINE_int32(max_num_refinement_iterations, 0, "Iterative refinement iterations");
DEFINE_bool(line_search, false, "Use a line search instead of trust region algorithm.");
DEFINE_double(subset_fraction, 0.2, "The fraction of residual blocks to use for the subset preconditioner.");


void SetLinearSolver(ceres::Solver::Options* options)
{
    CHECK(StringToLinearSolverType(CERES_GET_FLAG(FLAGS_linear_solver), &options->linear_solver_type));
    CHECK(StringToPreconditionerType(CERES_GET_FLAG(FLAGS_preconditioner),&options->preconditioner_type));
    CHECK(StringToSparseLinearAlgebraLibraryType(
        CERES_GET_FLAG(FLAGS_sparse_linear_algebra_library), 
        &options->sparse_linear_algebra_library_type));
    options->use_mixed_precision_solves = CERES_GET_FLAG(FLAGS_mixed_precision_solves);
    options->max_num_refinement_iterations = CERES_GET_FLAG(FLAGS_max_num_refinement_iterations);
}

void SetMinimizerOptions(ceres::Solver::Options* options)
{
    options->max_num_iterations = CERES_GET_FLAG(FLAGS_num_iterations);
    options->minimizer_progress_to_stdout = true;
    options->num_threads = CERES_GET_FLAG(FLAGS_num_threads);
    options->eta = CERES_GET_FLAG(FLAGS_eta);
    options->use_nonmonotonic_steps = CERES_GET_FLAG(FLAGS_nonmonotonic_steps);

    if (CERES_GET_FLAG(FLAGS_line_search)) {
        options->minimizer_type = ceres::LINE_SEARCH;
    }

    CHECK(StringToTrustRegionStrategyType(
        CERES_GET_FLAG(FLAGS_trust_region_strategy),
        &options->trust_region_strategy_type));
    CHECK(StringToDoglegType(CERES_GET_FLAG(FLAGS_dogleg), &options->dogleg_type));
    options->use_inner_iterations = CERES_GET_FLAG(FLAGS_inner_iterations);
}

void CreateProblem(double* solution, double* input, int width, int height, std::array<int, 4> roi, double lambda, ceres::Problem* problem)
{
    std::cout << "[INFO]: Creating the problem." << std::endl;

    // data consistency term
    ceres::LossFunction* data_loss = new ceres::HuberLoss(1);

    for (int h = roi[2]; h < roi[3]; h++)
    {
        for (int w = roi[0]; w < roi[1]; w++)
        {
            int index = h * width + w;
            if (!std::isnan(input[index]))
            {
                ceres::CostFunction* data_cost = new DataConsistencyCostFunction(input[index]);
                problem->AddResidualBlock(data_cost, data_loss, &(solution[index]));
            }
        }
    }
    std::cout << "[INFO]: Successfully added data term." << std::endl;

    // smoothness term
    ceres::LossFunction* smoothness_loss = new ceres::HuberLoss(1);

    for (int h = roi[2]; h < roi[3]; h++)
    {
        for (int w = roi[0]; w < roi[1]; w++)
        {
            std::vector<double*> pixels;
            std::vector<double> weights;

            int c_idx = width * h + w;
            int l_idx = width * h + w - 1;
            int r_idx = width * h + w + 1;
            int t_idx = width * (h - 1) + w;
            int b_idx = width * (h + 1) + w;

            pixels.push_back(&(solution[c_idx]));
            weights.push_back(0.0); // occupy the first element, will update later

            if (w != 0)
            {
                pixels.push_back(&(solution[l_idx]));
                weights.push_back(lambda / (abs(solution[c_idx] - solution[l_idx]) + 0.004)); // 1 / 255 ~= 0.004
            }

            if (w != width - 1)
            {
                pixels.push_back(&(solution[r_idx]));
                weights.push_back(lambda / (abs(solution[c_idx] - solution[r_idx]) + 0.004));
            }

            if (h != 0)
            {
                pixels.push_back(&(solution[t_idx]));
                weights.push_back(lambda / (abs(solution[c_idx] - solution[t_idx]) + 0.004));
            }

            if (h != height - 1)
            {
                pixels.push_back(&(solution[b_idx]));
                weights.push_back(lambda / (abs(solution[c_idx] - solution[b_idx]) + 0.004));
            }

            // update weight for cetral pixel
            for (int i = 1; i < weights.size(); i++)
            {
                weights[0] += weights[i];
            }

            ceres::CostFunction* smoothness_cost = new SmoothnessCostFunction(weights);
            problem->AddResidualBlock(smoothness_cost, smoothness_loss, pixels);
        }
    }
    std::cout << "[INFO]: Successfully added smoothness term." << std::endl;
}

void SolveProblem(ceres::Problem* problem, double* solution)
{
    std::cout << "[INFO]: Solving the problem..." << std::endl;
    ceres::Solver::Options options;
    SetMinimizerOptions(&options);
    SetLinearSolver(&options);
    options.function_tolerance = 1e-3;
    std::cout << "[INFO]: Successfully updated solver options." << std::endl;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

int main(int argc, char** argv)
{
    // get arguments
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    if (CERES_GET_FLAG(FLAGS_input).empty())
    {
        std::cerr << "[FAIL]: Please provide an image file name using -input.\n";
        return -1;
    }
    std::cout << "[INFO]: Successfully parsed arguments." << std::endl;

    // read the images
    int width = 0;
    int height = 0;
    double* image = LoadGrayscaleImage(CERES_GET_FLAG(FLAGS_input), width, height);

    if (height == 0)
    {
        std::cerr << "[FAIL]: Failed to load input image.\n";
        return -1;
    }
    std::cout << "[INFO]: Successfully loaded image of size (" << width << ", " << height << ")." << std::endl;

    // roi
    std::array<int, 4> roi = {0, width, 0, height}; // x_min, x_max, y_min, y_max
    if (CERES_GET_FLAG(FLAGS_x_min) != -1)
        roi[0] = CERES_GET_FLAG(FLAGS_x_min);
    if (CERES_GET_FLAG(FLAGS_x_max) != -1)
        roi[1] = CERES_GET_FLAG(FLAGS_x_max);
    if (CERES_GET_FLAG(FLAGS_y_min) != -1)
        roi[2] = CERES_GET_FLAG(FLAGS_y_min);
    if (CERES_GET_FLAG(FLAGS_y_max) != -1)
        roi[3] = CERES_GET_FLAG(FLAGS_y_max);

    // create solution space
    double* solution = (double*) malloc(sizeof(double) * width * height);
    memcpy(solution, image, sizeof(double) * width * height);
    std::cout << "[INFO]: Successfully created solution space." << std::endl;

    // solve problem
    ceres::Problem problem;
    CreateProblem(solution, image, width, height, roi, CERES_GET_FLAG(FLAGS_lambda), &problem);
    std::cout << "[INFO]: Successfully created the problem." << std::endl;
    SolveProblem(&problem, solution);
    std::cout << "[INFO]: Successfully solved the problem." << std::endl;

    // save image
    std::string output_path;
    if (CERES_GET_FLAG(FLAGS_output).empty())
        output_path = "./diffusion_output.png";
    else
        output_path = CERES_GET_FLAG(FLAGS_output);

    SaveGrayscaleImage(output_path, solution, width, height);
    std::cout << "[INFO]: Successfully save the output image." << std::endl;

    return 0;
}
