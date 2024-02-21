#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Function to load array from file
float * loadArrayFromFile(const char *filename, int &rows, int &cols)
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

int main()
{
    // Batch size
    int B = 6000;
    // Image pixels
    int L = 28 * 28;
    // Number of classes
    int C = 10;
    float *trainImages = loadArrayFromFile("train-images.idx3-ubyte_sample_True.6000.784.txt", B, L);
    float *trainLables = loadArrayFromFile("train-labels.idx1-ubyte_sample_True.6000.10.txt", B, C);
    if (!trainImages)
    {
        std::cerr << "Error loading array from file." << std::endl;
        return 1;
    }

    // Free the allocated memory
    delete[] trainImages;
    delete[] trainLables;
    return 0;
}