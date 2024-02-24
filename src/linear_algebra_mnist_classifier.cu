#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <tuple>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 3

// Function to load array from file
__host__ float *loadArrayFromFile(const char* filename, int &rows, int &cols)
{
    // Allocate memory for the array
    float *array = new float[rows * cols];

    // Open the file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file." << std::endl;
    }

    std::cout << "Opened file" << std::endl;
    // Read each line from the file
    std::string line;
    int index = 0;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        float value;
        while (iss >> value)
        {
            array[index] = value;
            index++;
        }
    }
    std::cout << "loaded " << index << " elements into array" << std::endl;

    // Close the file
    file.close();

    return array;
}

__host__ std::tuple<float *, float *> loadImageAndLabel(std::string dataset, int batch_size, int pixels_length, int class_num)
{
    // dataset is either "train" or "t10k"
    std::string image_data_path = dataset + "-images.idx3-ubyte_sample_True." + std::to_string(batch_size) + "." + std::to_string(pixels_length) + ".txt";
    std::string label_data_path = dataset + "-labels.idx1-ubyte_sample_True." + std::to_string(batch_size) + "." + std::to_string(class_num) + ".txt";
    float *host_image = loadArrayFromFile(image_data_path.c_str(), batch_size, pixels_length);
    float *host_label = loadArrayFromFile(label_data_path.c_str(), batch_size, class_num);
    return std::make_tuple(host_image, host_label);
}

// kernels for element-wise operations
__global__ void reluKernel(float *d_a, float *d_b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        d_b[i] = d_a[i] > 0 ? d_a[i] : 0;
    }
}

__global__ void reluDerivativeKernel(float *d_a, float *d_b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        d_b[i] = d_a[i] > 0 ? 1.0 : 0.0;
    }
}

__global__ void elementWiseNegativeDivideKernel(float *d_a, float *d_b, float *d_c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        d_c[i] = -d_a[i] / d_b[i];
    }
}

__global__ void elementWiseMultiplyKernel(float *d_a, float *d_b, float *d_c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        d_c[i] = d_a[i] * d_b[i];
    }
}

__global__ void logKernel(float *d_A, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        d_A[i] = logf(d_A[i]);
    }
}

__host__ void forwarPass(cublasHandle_t handle, float *d_x, float *d_w1, float *d_s, float *d_z, float *d_w2, float *d_p, int batch_size, int pixel_len, int hidden_size, int class_num)
{
    /*
     d_s = d_x * d_w1
     d_z = relu(d_s)
     d_p = d_z * d_w2
     */
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, hidden_size, pixel_len, &alpha, d_x, batch_size, d_w1, pixel_len, &beta, d_s, batch_size);
    reluKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_s, d_z, batch_size * hidden_size);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, class_num, hidden_size, &alpha, d_z, batch_size, d_w2, hidden_size, &beta, d_p, batch_size);
}

__host__ void backPropagate(cublasHandle_t handle, float *d_w1, float *d_dw1, float *d_w2, float *d_dw2, float *d_x, float *d_s, float *d_ds, float *d_z, float *d_dz, float *d_p, float *d_dp, float *d_y, int batch_size, int pixel_len, int hidden_size, int class_num, const float learning_rate)
{
    /*
    calculate d_dw2 and d_dw1
    d_dp = - d_y / d_p // element-wise divide
    d_dw2 = T(d_z) * d_dp
    d_dz = d_dp * T(d_w2)
    d_ds = d_dz x d_dz // element-wise product
    d_dw1 = T(d_x) * d_ds
    */
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    elementWiseNegativeDivideKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_y, d_p, d_dp, batch_size * class_num);

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, batch_size, class_num, &alpha, d_z, batch_size, d_dp, batch_size, &beta, d_dw2, hidden_size);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch_size, hidden_size, class_num, &alpha, d_dp, batch_size, d_w2, hidden_size, &beta, d_dz, batch_size);
    reluDerivativeKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_z, d_dz, batch_size * hidden_size);
    elementWiseMultiplyKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_dz, d_dz, d_ds, batch_size * hidden_size);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, pixel_len, hidden_size, batch_size, &alpha, d_x, batch_size, d_ds, batch_size, &beta, d_dw1, pixel_len);

    /*
    update d_w1 and d_w2
    d_w1 = d_w1 + lr * d_dw1
    d_w2 = d_w2 + lr * d_dw2
    */

    /*
    cublasSaxpy(handle, pixel_len * hidden_size, &learning_rate, d_dw1, 1, d_w1, 1);
    cublasSaxpy(handle, hidden_size * class_num, &learning_rate, d_dw2, 1, d_w2, 1);
    */
}

__host__ std::tuple<float *, float *> allocateHostMemory(int batch_size, int pixel_len, int hidden_size, int class_num)
{
    float *h_w1, *h_w2;
    size_t size_w1 = pixel_len * hidden_size * sizeof(float);
    size_t size_w2 = hidden_size * class_num * sizeof(float);
    cudaMallocHost(&h_w1, size_w1);
    cudaMallocHost(&h_w2, size_w2);
    return std::make_tuple(h_w1, h_w2);
}

__host__ std::tuple<float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *> allocateDeviceMemory(int batch_size, int pixel_len, int hidden_size, int class_num)
{
    // model weights and their derivative matrix
    float *d_w1, *d_w2, *d_dw1, *d_dw2;
    size_t size_w1 = pixel_len * hidden_size * sizeof(float);
    size_t size_w2 = hidden_size * class_num * sizeof(float);

    cudaMalloc(&d_w1, size_w1);
    cudaMalloc(&d_w2, size_w2);
    cudaMalloc(&d_dw1, size_w1);
    cudaMalloc(&d_dw2, size_w2);

    // input, output, intermediate matrix and their derivative
    float *d_x, *d_s, *d_ds, *d_z, *d_dz, *d_p, *d_dp, *d_y;
    size_t size_x = batch_size * pixel_len * sizeof(float);
    size_t size_s = batch_size * hidden_size * sizeof(float);
    size_t size_z = batch_size * hidden_size * sizeof(float);
    size_t size_p = batch_size * class_num * sizeof(float);
    size_t size_y = batch_size * class_num * sizeof(float);

    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_s, size_s);
    cudaMalloc(&d_ds, size_s);
    cudaMalloc(&d_z, size_z);
    cudaMalloc(&d_dz, size_z);
    cudaMalloc(&d_p, size_p);
    cudaMalloc(&d_dp, size_p);
    cudaMalloc(&d_y, size_y);

    return std::make_tuple(d_w1, d_dw1, d_w2, d_dw2, d_x, d_s, d_ds, d_z, d_dz, d_p, d_dp, d_y);
}

__host__ std::tuple<float *, float *, float *, float *> allocateTestDeviceMemory(int test_batch_size, int pixel_len, int hidden_size, int class_num)
{
    // input, output, intermediate matrix and their derivative
    float *d_x, *d_s, *d_z, *d_p;
    size_t size_x = test_batch_size * pixel_len * sizeof(float);
    size_t size_s = test_batch_size * hidden_size * sizeof(float);
    size_t size_z = test_batch_size * hidden_size * sizeof(float);
    size_t size_p = test_batch_size * class_num * sizeof(float);

    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_s, size_s);
    cudaMalloc(&d_z, size_z);
    cudaMalloc(&d_p, size_p);

    return std::make_tuple(d_x, d_s, d_z, d_p);
}

void xavier_weight_init(int n, float* h_w, int s) {
    double lower_bound = -std::sqrt(1.0 / n);
    double upper_bound = std::sqrt(1.0 / n);

    // Create a random number generator
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

    // Generate a random number between -sqrt(1/n) and sqrt(1/n)
    for (int i = 0; i < s; i++) {
        h_w[i] = dist(eng);
    }
}

__host__ float test_accuracy(float *h_y, float *h_p, int test_batch_size, int class_num) {
    int total_num = test_batch_size;
    int correct_num = 0;
    for (int i = 0; i < test_batch_size; i++)
    {
        int max_prob_index = i * class_num;
        float max_prob = 0.0;
        for (int j = 0; j < class_num; j++)
        {
            if (h_p[i * class_num + j] > max_prob) {
                max_prob = h_p[i * class_num + j];
                max_prob_index = i * class_num + j;
            }
        }
        // check if the respective element in ground-truth is 1.
        if (h_y[max_prob_index] > 0.999) {
            correct_num++;
        }
    }
    float accuracy = (float) correct_num / (float) total_num;
    printf("Accuracy of the model is: %.3f\n", accuracy);
    return accuracy;
}

int main()
{
    // train batch size
    int batch_size = 6000;
    // test batch size
    int test_batch_size = 1000;
    // Image pixels
    int pixel_len = 28 * 28;
    // Number of classes
    int class_num = 10;
    // Hidden size of first MLP
    int hidden_size = 512;
    // number of epochs
    int epoch_num = 3;
    const float learning_rate = 0.001;

    // create handle for cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << "load train and test files" << std::endl;
    auto [h_train_image, h_train_label] = loadImageAndLabel("train", batch_size, pixel_len, class_num);
    auto [h_test_image, h_test_label] = loadImageAndLabel("t10k", test_batch_size, pixel_len, class_num);

    if (!h_train_image || !h_train_label || !h_test_image || !h_test_label)
    {
        std::cerr << "Error loading array from file." << std::endl;
        return 1;
    }

    std::cout << "allocate host memory for training model" << std::endl;
    // allocate host memory for model training
    auto [h_w1, h_w2]  = allocateHostMemory(batch_size, pixel_len, hidden_size, class_num);

    std::cout << "allocate device memory for training model" << std::endl;
    // allocate device memory for model training
    auto [d_w1, d_dw1, d_w2, d_dw2, d_x, d_s, d_ds, d_z, d_dz, d_p, d_dp, d_y] = allocateDeviceMemory(batch_size, pixel_len, hidden_size, class_num);

    std::cout << "initialize weights and copy to device" << std::endl;
    // initialize weights and copy to device
    xavier_weight_init(pixel_len, h_w1, pixel_len * hidden_size);
    xavier_weight_init(hidden_size, h_w2, hidden_size * class_num);
    cudaMemcpy(d_w1, h_w1, pixel_len * hidden_size *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, hidden_size * class_num * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "copy images and labels to device" << std::endl;
    // copy images and labels to device
    cudaMemcpy(d_x, h_train_image, batch_size * pixel_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_train_label, batch_size * class_num * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "train the model" << std::endl;
    // Train the model
    for (int i = 0; i < epoch_num; i++)
    {
        std::cout << "epoch: " << i << " forwarPass" << std::endl;
        forwarPass(handle, d_x, d_w1, d_s, d_z, d_w2, d_p, batch_size, pixel_len, hidden_size, class_num);
        float *h_p = new float[batch_size*class_num];
        cudaMemcpy(h_p, d_p, batch_size*class_num * sizeof(float), cudaMemcpyDeviceToHost);
        for(int j = 0; j < class_num; j++) {
            printf("%.3f ", h_p[j]);
        }
        std::cout << std::endl;
        delete[] h_p;
        std::cout << "epoch: " << i << " backPropagate" << std::endl;
        backPropagate( handle, d_w1, d_dw1, d_w2, d_dw2, d_x, d_s, d_ds, d_z, d_dz, d_p, d_dp, d_y, batch_size, pixel_len, hidden_size, class_num, -learning_rate);
        cudaDeviceSynchronize();
    }
    
    std::cout << "free the allocated memory for training" << std::endl;
    // Free the allocated memory for training
    delete[] h_train_image;
    delete[] h_train_label;
    cudaFree(d_dw1);
    cudaFree(d_dw2);
    cudaFree(d_x);
    cudaFree(d_s);
    cudaFree(d_ds);
    cudaFree(d_z);
    cudaFree(d_dz);
    cudaFree(d_p);
    cudaFree(d_dp);
    cudaFree(d_y);

    std::cout << "test model" << std::endl;
    // Test the model
    auto [d_test_x, d_test_s, d_test_z, d_test_p] = allocateTestDeviceMemory(test_batch_size, pixel_len, hidden_size, class_num);
    float *h_test_p = new float[test_batch_size * class_num];
    std::cout << "forward pass" << std::endl;
    forwarPass(handle, d_test_x, d_w1, d_test_s, d_test_z, d_w2, d_test_p, test_batch_size, pixel_len, hidden_size, class_num);
    cudaMemcpy(h_test_p, d_test_p, test_batch_size * class_num * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "test accuracy" << std::endl;
    test_accuracy(h_test_label, h_test_p, test_batch_size, class_num);

    std::cout << "free memory allocated for model weights and testing" << std::endl;
    // Free memory allocated for model weights and testing 
    cublasDestroy(handle);
    cudaFree(d_w1);
    cudaFree(d_w2);

    delete[] h_test_image;
    delete[] h_test_label;
    delete[] h_test_p;
    cudaFreeHost(h_w1);
    cudaFreeHost(h_w2);
    cudaFree(d_test_x);
    cudaFree(d_test_s);
    cudaFree(d_test_z);
    cudaFree(d_test_p);
    return 0;
}