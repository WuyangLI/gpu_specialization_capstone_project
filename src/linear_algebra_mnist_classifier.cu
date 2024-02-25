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
    // normalize image to [0, 1] interval
    for(int i = 0; i < batch_size*pixels_length; i++) {
        host_image[i] = host_image[i] / 255.0f;
    }
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
    float epsilon = 1.0e-15;
    for (int i = index; i < n; i += stride)
    {
        d_c[i] = - d_a[i] / (d_b[i] + epsilon);
    }
}

__global__ void softmaxKernel(float *d_a, float *d_c, int row_num, int col_num)
{
    /*
    This is not an efficient implementation of softmax
    for numeric stability, we calcualte softmax in the following way:
        exp(pij - max_row_i) / sum(exp(pi* - max_row_i))
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n = row_num * col_num;
    for (int i = index; i < n; i += stride)
    {
        int row = i / col_num;
        double exp_sum = 0.0;
        float max_row_val = d_a[row * col_num];
        for (int j = row * col_num; j < row * col_num + col_num; j++) {
            max_row_val = max(max_row_val, d_a[j]);
        }
        for (int j = row * col_num; j < row * col_num + col_num; j++) {
            exp_sum += exp(d_a[j] - max_row_val);
        }
        d_c[i] = (float) (exp(d_a[i] - max_row_val) / exp_sum);
    }
}

__global__ void softmaxDerivativeKernel(float *d_sm, float *d_dsm, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        d_dsm[i] = d_sm[i] - d_sm[i] * d_sm[i];
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

__global__ void logKernel(float *d_a, float *d_b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float epsilon = 1.0e-15;
    for (int i = index; i < n; i += stride)
    {
        d_b[i] = log2f(d_a[i] + epsilon);
    }
}

__global__ void clipGradientKernel(float *d_g,  float min_gradient, float max_gradient, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        if (d_g[i] < min_gradient) {
            d_g[i] = min_gradient;
        }
        else if (d_g[i] > max_gradient) {
            d_g[i] = max_gradient;
        }
        else {
            d_g[i] = d_g[i];
        }
    }   
}

__host__ void forwardPass(cublasHandle_t handle, float *d_x, float *d_w1, float *d_s, float *d_z, float *d_w2, float *d_f, float *d_p, int batch_size, int pixel_len, int hidden_size, int class_num)
{
    /*
     d_s = d_x * d_w1
     d_z = relu(d_s)
     d_f = d_z * d_w2
     d_p = softmax(d_f)
     */
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status;
    cudaError_t cudaError;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, hidden_size, pixel_len, &alpha, d_x, batch_size, d_w1, pixel_len, &beta, d_s, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Sgemm failed when calculating d_s" << std::endl;
        cudaError = cudaGetLastError();
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }
    reluKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_s, d_z, batch_size * hidden_size);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel error from reluKernel: " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, class_num, hidden_size, &alpha, d_z, batch_size, d_w2, hidden_size, &beta, d_f, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Sgemm failed when calculating d_f" << std::endl;
        return;
    }
    softmaxKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_f, d_p, batch_size, class_num);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel error from softmaxKernel: " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }
}

__host__ void backPropagate(cublasHandle_t handle, float *d_w1, float *d_dw1, float *d_w2, float *d_dw2, float *d_x, float *d_s, float *d_ds, float *d_z, float *d_dz, float *d_relu, float *d_f, float *d_df, float *d_p, float *d_dp, float *d_softmax, float *d_y, int batch_size, int pixel_len, int hidden_size, int class_num, const float learning_rate)
{
    /*
    */
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status;
    // dp = - np.divide(y, p + epsilon)
    elementWiseNegativeDivideKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_y, d_p, d_dp, batch_size * class_num);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel error when calculating d_dp: " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }
    // df = np.multiply(dp, p - np.multiply(p, p))
    softmaxDerivativeKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_p, d_softmax, batch_size * class_num);
    elementWiseMultiplyKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_dp, d_softmax, d_df, batch_size * class_num);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel error when calculating d_df: " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }

    // dw2 = np.dot(np.transpose(z), df)
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, class_num, batch_size, &alpha, d_z, batch_size, d_df, batch_size, &beta, d_dw2, hidden_size);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Sgemm failed when calculating d_dw2" << std::endl;
        return;
    }

    // dz = np.dot(df, np.transpose(w2))
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, batch_size, hidden_size, class_num, &alpha, d_df, batch_size, d_w2, hidden_size, &beta, d_dz, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Sgemm failed when calculating d_dz" << std::endl;
        return;
    }

    // ds = np.multiply(dz, (s > 0).astype(float))
    reluDerivativeKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_s, d_relu, batch_size * hidden_size);
    elementWiseMultiplyKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_dz, d_relu, d_ds, batch_size * hidden_size);

    // dw1 = np.dot(np.transpose(x), ds)
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, pixel_len, hidden_size, batch_size, &alpha, d_x, batch_size, d_ds, batch_size, &beta, d_dw1, pixel_len);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Sgemm failed when calculating d_dw1" << std::endl;
        return;
    }

    /*
    update d_w1 and d_w2
    d_w1 = d_w1 - lr * d_dw1
    d_w2 = d_w2 - lr * d_dw2
    */

    // clip the gradients of w1 and w2 to [-1, 1]
    clipGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_dw1, -1.0f, 1.0f, pixel_len * hidden_size);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel error when clipping gradients of w1 : " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }
    clipGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_dw2, -1.0f, 1.0f, hidden_size * class_num);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel error when clipping gradients of w2 : " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }
    const float lr = - learning_rate;
    status = cublasSaxpy(handle, pixel_len * hidden_size, &lr, d_dw1, 1, d_w1, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Saxpy failed when updating d_w1" << std::endl;
        return;
    }
    status = cublasSaxpy(handle, hidden_size * class_num, &lr, d_dw2, 1, d_w2, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Saxpy failed when updating d_w2" << std::endl;
        return;
    }
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

__host__ std::tuple<float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *> allocateDeviceMemory(int batch_size, int pixel_len, int hidden_size, int class_num)
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
    float *d_x, *d_s, *d_ds, *d_z, *d_dz, *d_relu, *d_f, *d_df, *d_p, *d_dp, *d_softmax, *d_y;
    size_t size_x = batch_size * pixel_len * sizeof(float);
    size_t size_s = batch_size * hidden_size * sizeof(float);
    size_t size_z = batch_size * hidden_size * sizeof(float);
    size_t size_f = batch_size * class_num * sizeof(float);
    size_t size_p = batch_size * class_num * sizeof(float);
    size_t size_y = batch_size * class_num * sizeof(float);

    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_s, size_s);
    cudaMalloc(&d_ds, size_s);
    cudaMalloc(&d_z, size_z);
    cudaMalloc(&d_dz, size_z);
    cudaMalloc(&d_relu, size_z);
    cudaMalloc(&d_f, size_f);
    cudaMalloc(&d_df, size_f);
    cudaMalloc(&d_p, size_p);
    cudaMalloc(&d_dp, size_p);
    cudaMalloc(&d_softmax, size_p);
    cudaMalloc(&d_y, size_y);

    return std::make_tuple(d_w1, d_dw1, d_w2, d_dw2, d_x, d_s, d_ds, d_z, d_dz, d_relu, d_f, d_df, d_p, d_dp, d_softmax, d_y);
}

__host__ std::tuple<float *, float *, float *, float *, float *> allocateTestDeviceMemory(int test_batch_size, int pixel_len, int hidden_size, int class_num)
{
    // input, output, intermediate matrix and their derivative
    float *d_x, *d_s, *d_z, *d_f, *d_p;
    size_t size_x = test_batch_size * pixel_len * sizeof(float);
    size_t size_s = test_batch_size * hidden_size * sizeof(float);
    size_t size_z = test_batch_size * hidden_size * sizeof(float);
    size_t size_f = test_batch_size * class_num * sizeof(float);
    size_t size_p = test_batch_size * class_num * sizeof(float);

    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_s, size_s);
    cudaMalloc(&d_z, size_z);
    cudaMalloc(&d_f, size_f);
    cudaMalloc(&d_p, size_p);

    return std::make_tuple(d_x, d_s, d_z, d_f, d_p);
}

void xavierWeightInit(int n, float* h_w, int s) {
    double lower_bound = -std::sqrt(1.0 / (float) n);
    double upper_bound = std::sqrt(1.0 / (float) n);

    // Create a random number generator
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

    // Generate a random number between -sqrt(1/n) and sqrt(1/n)
    for (int i = 0; i < s; i++) {
        h_w[i] = dist(eng);
    }
}

__host__ float calculateLoss(float *d_p, float *d_y, int batch_size, int class_num) {
    float *d_loss;
    size_t size_l = batch_size * class_num * sizeof(float);
    float *h_loss = new float[batch_size * class_num];
    float *d_log_p = new float[batch_size * class_num];

    cudaMalloc(&d_log_p, size_l);
    cudaMalloc(&d_loss, size_l);

    logKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_p, d_log_p, batch_size * class_num);
    elementWiseMultiplyKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_y, d_log_p, d_loss, batch_size * class_num);
    cudaMemcpy(h_loss, d_loss, size_l, cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    cudaFree(d_log_p);

    float loss = 0.0f;
    for (int i = 0; i < batch_size * class_num; i++) {
        loss += (-1.0f / batch_size) * h_loss[i];
    }
    delete[] h_loss;
    return loss;
}

__host__ float testAccuracy(float *h_y, float *h_p, int test_batch_size, int class_num) {
    int total_num = test_batch_size;
    int correct_num = 0;
    for (int i = 0; i < test_batch_size; i++)
    {
        int max_prob_index = i * class_num;
        int max_y_index = i * class_num;
        float max_prob = 0.0;
        for (int j = 0; j < class_num; j++)
        {
            if (h_p[i * class_num + j] > max_prob) {
                max_prob = h_p[i * class_num + j];
                max_prob_index = j;
            }

            if (h_y[i * class_num + j] > 0.999) {
                max_y_index = j;
            }
        }
        // check if the respective element in ground-truth is 1.
        if (max_prob_index == max_y_index) {
            correct_num++;
        }
    }
    float accuracy = (float) correct_num / (float) total_num;
    printf("Accuracy of the model is: %.3f\n", accuracy);
    return accuracy;
}

void debug_print_a_matrix(float *d_m, int dim1, int dim2) {
    float *h_m = new float[dim1 * dim2];
    cudaMemcpy(h_m, d_m, dim1 * dim2 * sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < dim2; j++) {
        printf("%.3f ", h_m[j]);
    }
    std::cout << std::endl;
    delete[] h_m;
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
    int epoch_num = 10;
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
    auto [d_w1, d_dw1, d_w2, d_dw2, d_x, d_s, d_ds, d_z, d_dz, d_relu, d_f, d_df, d_p, d_dp, d_softmax, d_y] = allocateDeviceMemory(batch_size, pixel_len, hidden_size, class_num);

    std::cout << "initialize weights and copy to device" << std::endl;
    // initialize weights and copy to device
    xavierWeightInit(pixel_len, h_w1, pixel_len * hidden_size);
    xavierWeightInit(hidden_size, h_w2, hidden_size * class_num);
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
        std::cout << "epoch: " << i << " forwardPass" << std::endl;
        forwardPass(handle, d_x, d_w1, d_s, d_z, d_w2, d_f, d_p, batch_size, pixel_len, hidden_size, class_num);
        float loss = calculateLoss(d_p, d_y, batch_size, class_num);
        printf("epoech %d loss is %.3f \n", i, loss);
        std::cout << "epoch: " << i << " backPropagate" << std::endl;
        backPropagate(handle, d_w1, d_dw1, d_w2, d_dw2, d_x, d_s, d_ds, d_z, d_dz, d_relu, d_f, d_df, d_p, d_dp, d_softmax, d_y, batch_size, pixel_len, hidden_size, class_num, learning_rate);
        cudaDeviceSynchronize();
        std::cout << std::endl;
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
    cudaFree(d_f);
    cudaFree(d_df);
    cudaFree(d_p);
    cudaFree(d_dp);
    cudaFree(d_relu);
    cudaFree(d_softmax);
    cudaFree(d_y);

    std::cout << "test model" << std::endl;
    // Test the model
    auto [d_test_x, d_test_s, d_test_z, d_test_f, d_test_p] = allocateTestDeviceMemory(test_batch_size, pixel_len, hidden_size, class_num);
    float *h_test_p = new float[test_batch_size * class_num];
    std::cout << "forward pass" << std::endl;
    forwardPass(handle, d_test_x, d_w1, d_test_s, d_test_z, d_w2, d_test_f, d_test_p, test_batch_size, pixel_len, hidden_size, class_num);
    cudaMemcpy(h_test_p, d_test_p, test_batch_size * class_num * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "test accuracy" << std::endl;
    testAccuracy(h_test_label, h_test_p, test_batch_size, class_num);

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
    cudaFree(d_test_f);
    cudaFree(d_test_p);
    return 0;
}