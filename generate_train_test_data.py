import struct
from array import array
from os.path import join
import argparse

def read_images_labels(images_filepath, labels_filepath, sample=False):        
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels_data = array("B", file.read())
        labels = []

        
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
        images = []

    for i in range(size):
        label = int(labels_data[i])
        if sample and i % 10 != 0:
            continue
        labels.append([0.0] * label + [1.0] + [0.0] * (9 - label))   
        img = image_data[i * rows * cols:(i + 1) * rows * cols]
        images.append(img)            

    print(f"images total number {len(images)}")
    print(f"images dim {len(images[0])}")
    print(f"labels total number {len(labels)}")
    print(f"labels dim {len(labels[0])}")
    # Save the array to a text file
    def save_to_file(fp, array_to_save):
        with open(fp, 'w') as file:
            for row in array_to_save:
                file.write(' '.join(map(str, row)) + '\n')
            
    save_to_file(f"{images_filepath}_sample_{sample}.{len(images)}.{len(images[0])}.txt", images)
    save_to_file(f"{labels_filepath}_sample_{sample}.{len(labels)}.{len(labels[0])}.txt", labels)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process image and label file paths")
    parser.add_argument("--image-file-path", required=True, help="Path to the image file")
    parser.add_argument("--label-file-path", required=True, help="Path to the label file")
    args = parser.parse_args()

    # Call the main function with provided arguments
    read_images_labels(args.image_file_path, args.label_file_path)