#include <cmath>
#include <array>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "ceres/cubic_interpolation.h"
#include "opencv4/opencv2/imgcodecs.hpp"

#include "utils.h"
#include "data_consistency.h"
#include "spatial_smoothness.h"


// input variables
DEFINE_string(left, "", "Left Image of a rgb-d camera.");
DEFINE_string(right, "", "Right image of a rbg-d camera.");
DEFINE_string(pcd, "", "Input pcd file. Assumed 2d with each point matching each pixel of left image.");
DEFINE_string(camera, "", "Camera intrinsic for left and right camera as well as the transformation from left to right.");
DEFINE_string(output, "", "File to which the output image should be written");
DEFINE_double(lambda, 0.0, "Weight on spatial smoothness term. Default: 1.0");
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
DEFINE_int32(num_iterations, 50, "Number of iterations.");
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

double* InverseZ(double* pcd, int width, int height)
{
    double* z_inv = (double*)malloc(width * height * sizeof(double));

    for (int i = 0; i < width * height; i++)
    {
        double z = pcd[3 * i + 2];
        z_inv[i] = 1 / z;
    }
    
    return z_inv;
}

double* Reprojection(double* z_inv, int width, int height, double fx, double fy, double cx, double cy)
{
    double* pcd = (double*)malloc(3 * width * height * sizeof(double));

    for (unsigned v = 0; v < height; v++)
    {
        for (unsigned u = 0; u < width; u++)
        {
            unsigned i = v * width + u;
            pcd[3 * i + 0] = (u - cx) / fx / z_inv[i];
            pcd[3 * i + 1] = (v - cy) / fy / z_inv[i];
            pcd[3 * i + 2] = 1 / z_inv[i];
        }
    }

    return pcd;
}

void AddDataConsistencyTerms(ceres::Problem* problem, double* solution, double* depth, int width, int height, const std::array<int, 4>& roi)
{
    ceres::LossFunction* data_loss = new ceres::HuberLoss(1);

    for (int v = roi[2]; v < roi[3]; v++)
    {
        for (int u = roi[0]; u < roi[1]; u++)
        {
            int index = v * width + u;
            if (!std::isnan(depth[index]))
            {
                ceres::CostFunction* data_cost = new DataConsistencyCostFunction(depth[index]);
                problem->AddResidualBlock(data_cost, data_loss, &(solution[index]));
            }
        }
    }
}

void AddSmoothnessTerms(ceres::Problem* problem, double* solution, double* img_l, int width, int height, const std::array<int, 4>& roi, double lambda)
{
    ceres::LossFunction* smoothness_loss = new ceres::HuberLoss(1);

    for (int v = roi[2]; v < roi[3]; v++)
    {
        for (int u = roi[0]; u < roi[1]; u++)
        {
            std::vector<double*> pixels;
            std::vector<double> weights;

            int c_idx = width *  v      + u;
            int l_idx = width *  v      + u - 1;
            int r_idx = width *  v      + u + 1;
            int t_idx = width * (v - 1) + u;
            int b_idx = width * (v + 1) + u;

            pixels.push_back(&(solution[c_idx]));
            weights.push_back(0.0); // occupy the first element, will update later

            if (u != 0)
            {
                pixels.push_back(&(solution[l_idx]));
                weights.push_back(lambda / (abs(img_l[c_idx] - img_l[l_idx]) + 0.004)); // 1 / 255 ~= 0.004
            }

            if (u != width - 1)
            {
                pixels.push_back(&(solution[r_idx]));
                weights.push_back(lambda / (abs(img_l[c_idx] - img_l[r_idx]) + 0.004));
            }

            if (v != 0)
            {
                pixels.push_back(&(solution[t_idx]));
                weights.push_back(lambda / (abs(img_l[c_idx] - img_l[t_idx]) + 0.004));
            }

            if (v != height - 1)
            {
                pixels.push_back(&(solution[b_idx]));
                weights.push_back(lambda / (abs(img_l[c_idx] - img_l[b_idx]) + 0.004));
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
    std::cout << summary.FullReport() << std::endl;
}

int main(int argc, char** argv)
{
    // get arguments
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    if (CERES_GET_FLAG(FLAGS_left).empty())
    {
        std::cout << "[INFO]: Please provide left image using -left." << std::endl;
        return -1;
    }
    if (CERES_GET_FLAG(FLAGS_right).empty())
    {
        std::cout << "[INFO]: Please provide right image using -right." << std::endl;
        return -1;
    }
    if (CERES_GET_FLAG(FLAGS_pcd).empty())
    {
        std::cout << "[INFO]: Please provide organized point cloud using -pcd." << std::endl;
        return -1;
    }
    if (CERES_GET_FLAG(FLAGS_camera).empty())
    {
        std::cout << "[INFO]: Please provide camera information using -camera." << std::endl;
        return -1;
    }
    std::cout << "[INFO]: Successfully parsed arguments." << std::endl;

    int width_l = 0, width_r = 0, width_d = 0;
    int height_l = 0, height_r = 0, height_d = 0;

    // read images
    double* img_l = LoadGrayscaleImage(CERES_GET_FLAG(FLAGS_left), width_l, height_l);
    double* img_r = LoadGrayscaleImage(CERES_GET_FLAG(FLAGS_right), width_r, height_r);
    ceres::Grid2D<double> grid_r(img_r, 0, height_r, 0, width_r);
    ceres::BiCubicInterpolator<ceres::Grid2D<double>> right(grid_r);

    // read point cloud
    double* pcd = LoadPointCloud(CERES_GET_FLAG(FLAGS_pcd), width_d, height_d);
    std::cout << "[INFO]: Successfully loaded point cloud." << std::endl;

    if (width_d != width_l || height_d != height_l)
    {
        std::cout << "[FAIL]: Input point cloud (" << width_d << " x " << height_d << ") and left image (" << width_l << " x " << height_l << ") has different size." << std::endl;
        return -1;
    }

    // convert to inverse depth
    double* z_inv = InverseZ(pcd, width_d, height_d);

    // roi
    std::array<int, 4> roi = {0, width_l, 0, height_l}; // x_min, x_max, y_min, y_max
    if (CERES_GET_FLAG(FLAGS_x_min) != -1)
        roi[0] = CERES_GET_FLAG(FLAGS_x_min);
    if (CERES_GET_FLAG(FLAGS_x_max) != -1)
        roi[1] = CERES_GET_FLAG(FLAGS_x_max);
    if (CERES_GET_FLAG(FLAGS_y_min) != -1)
        roi[2] = CERES_GET_FLAG(FLAGS_y_min);
    if (CERES_GET_FLAG(FLAGS_y_max) != -1)
        roi[3] = CERES_GET_FLAG(FLAGS_y_max);

    // camera parameters
    double b;
    double T[16];
    Camera cam_l(1.0, 1.0, 0.0, 0.0);
    Camera cam_r(1.0, 1.0, 0.0, 0.0);
    ReadCameraInfo(CERES_GET_FLAG(FLAGS_camera), cam_l.fx, cam_l.fy, cam_l.cx, cam_l.cy, cam_r.fx, cam_r.fy, cam_r.cx, cam_r.cy, b, T);

    // allocate space for solution
    double* solution = (double*) malloc(width_d * height_d * sizeof(double));
    memcpy(solution, z_inv, width_d * height_d * sizeof(double));
    
    for (int i = 0; i < width_d * height_d; i++)
    {
        if (std::isnan(z_inv[i]))
            solution[i] = 0.0021;
        else
            solution[i] = z_inv[i];
    }

    std::cout << "[INFO]: Successfully created solution space." << std::endl;

    // create problem
    ceres::Problem problem;
    AddDataConsistencyTerms(&problem, solution, z_inv, width_d, height_d, roi);
    
    else
        std::cout << "[INFO]: Photometric consistency term is not used." << std::endl;
    
    if (CERES_GET_FLAG(FLAGS_lambda) != 0.0)
    {
        std::cout << "[INFO]: Using smoothness term." << std::endl;
        AddSmoothnessTerms(&problem, solution, img_l, width_d, height_d, roi, CERES_GET_FLAG(FLAGS_lambda));
    }
    else
        std::cout << "[INFO]: Smoothness term is not used." << std::endl;
    
    std::cout << "[INFO]: Successfully create the problem." << std::endl;

    // solve problem
    SolveProblem(&problem, solution);
    std::cout << "[INFO]: Successfully solved the problem." << std::endl;
    double* pcd_o = Reprojection(solution, width_d, height_d, cam_l.fx, cam_l.fy, cam_l.cx, cam_l.cy);
    
    // save output
    std::string output_path;
    if (CERES_GET_FLAG(FLAGS_output).empty())
        output_path = "depth_refinment_output.png";
    else
        output_path = CERES_GET_FLAG(FLAGS_output);

    SavePointCloud("output.pcd", pcd_o, width_d, height_d);

    return 0;
}