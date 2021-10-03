#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <iostream>

#include "opencv4/opencv2/imgcodecs.hpp"


double* LoadGrayscaleImage(const std::string& file_path, int& width, int& height)
{
    cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);

    width = img.cols;
    height = img.rows;
    double* intensity = (double*)malloc(sizeof(double) * width * height);

    for (int h = 0; h < height; h++)
    {
        u_char* p = img.ptr<u_char>(h);
        for (int w = 0; w < width; w++)
        {
            unsigned int val = static_cast<unsigned int>(p[w]);
            intensity[h * width + w] = (float)val / 255.0f;
        }
    }

    return intensity;
}

void SaveGrayscaleImage(const std::string& file_path, double* output, int width, int height)
{
    cv::Mat image = cv::Mat::zeros(cv::Size(width, height),CV_8UC1);

    for (int h = 0; h < height; h++)
    {
        u_char* p = image.ptr<u_char>(h);
        for (int w = 0; w < width; w++)
        {
            unsigned int val = static_cast<unsigned int>(output[h * width + w] * 256.0f);
            p[w] = static_cast<u_char>(val);
        }
    }

    cv::imwrite(file_path, image);
}

double* LoadPointCloud(const std::string& file_path, int& width, int& height)
{
    std::ifstream stream(file_path.c_str(), std::ifstream::binary);

    if (!stream.is_open())
    {
        return nullptr;
    }

    while(true)
    {
        char char_arr[256];
        stream.getline(char_arr, 256);
        std::string line(char_arr);

        if (line.find("WIDTH") != std::string::npos)
        {
            int idx = line.find_last_of(" ");
            std::string str_width = line.substr(idx, line.size());
            width = stoi(str_width);
        }
        
        if (line.find("HEIGHT") != std::string::npos)
        {
            int idx = line.find_last_of(" ");
            std::string str_height = line.substr(idx, line.size());
            height = stoi(str_height);

        }

        if (line.find("DATA") != std::string::npos)
            break;
    };

    double* points = reinterpret_cast<double*>(malloc(3 * width * height * sizeof(double)));
    
    for (int i = 0; i < width * height; i++)
    {
        char buffer[16];
        stream.read(buffer, 16);

        float x = *reinterpret_cast<float*>(&buffer[0]);
        float y = *reinterpret_cast<float*>(&buffer[4]);
        float z = *reinterpret_cast<float*>(&buffer[8]);

        points[3 * i + 0] = static_cast<double>(x);
        points[3 * i + 1] = static_cast<double>(y);
        points[3 * i + 2] = static_cast<double>(z);
    }

    return points;
}

void ReadCameraInfo(const std::string& file_path, double& fx_l, double& fy_l, double& cx_l, double& cy_l, double& fx_r, double& fy_r, double& cx_r, double& cy_r, double& b, double* T)
{
    std::ifstream stream(file_path, std::ifstream::in);

    if (!stream.is_open())
    {
        std::cout << "[FAIL]: Failed to open camera file." << std::endl;
        return;
    }

    char char_arr[512];
    while (stream.getline(char_arr, 512))
    {
        if (stream.bad() || stream.eof())
            break;

        std::string line(char_arr);
        int idx = line.find_first_of("=");
        std::string key = line.substr(0, idx);
        std::string value = line.substr(idx + 1, line.length());

        if      (key == "fx_l") fx_l = stod(value);
        else if (key == "fy_l") fy_l = stod(value);
        else if (key == "cx_l") cx_l = stod(value);
        else if (key == "cy_l") cy_l = stod(value);
        else if (key == "fx_r") fx_r = stod(value);
        else if (key == "fy_r") fy_r = stod(value);
        else if (key == "cx_r") cx_r = stod(value);
        else if (key == "cy_r") cy_r = stod(value);
        else if (key == "b") b = stod(value);
        else if (key == "T")
        {
            value = value.substr(1, value.length() - 2);
            for (int i = 0; i < 16; i++)
            {
                // std::cout << value << std::endl;
                int pos = value.find_first_of(",");
                std::string num = value.substr(0, pos);
                T[i] = stod(num);
                value.erase(0, pos + 1);
            }
        }
        else 
            std::cout << "[WARN]: Unrecognized key '" << key << "' found." << std::endl;
    }
}

void SavePointCloudAsPLY(const std::string& file_path, double* pcd, int width, int height)
{
    std::ofstream stream(file_path.c_str(), std::ofstream::out);

    // write header
    stream << "ply\n";
    stream << "format ascii 1.0\n";
    stream << "comment This file contains point cloud.\n";
    stream << "element vertex " << width * height << "\n";
    stream << "property float x\n";
    stream << "property float y\n";
    stream << "property float z\n";
    stream << "end_header\n";

    // write points
    for (int i = 0; i < width * height; i++)
    {
        double x = pcd[3 * i + 0];
        double y = pcd[3 * i + 1];
        double z = pcd[3 * i + 2];
        stream << x << " " << y << " " << z << "\n";
    }

    stream.close();
}

void SavePointCloud(const std::string& file_path, double* pcd, int width, int height)
{
    std::ofstream stream(file_path.c_str(), std::ofstream::out);

    // write header
    stream << "VERSION 0.7\n";
    stream << "FIELDS x y z\n";
    stream << "SIZE 4 4 4\n";
    stream << "TYPE F F F\n";
    stream << "COUNT 1 1 1\n";
    stream << "WIDTH " << width << "\n";
    stream << "HEIGHT " << height << "\n";
    stream << "VIEWPOINT 0 0 0 1 0 0 0\n"; 
    stream << "POINTS " << width * height << "\n";
    stream << "DATA ascii" << "\n";

    // write points
    for (int i = 0; i < width * height; i++)
    {
        float x = static_cast<float>(pcd[3 * i + 0]);
        float y = static_cast<float>(pcd[3 * i + 1]);
        float z = static_cast<float>(pcd[3 * i + 2]);
        stream << x << " " << y << " " << z << "\n";
    }
}