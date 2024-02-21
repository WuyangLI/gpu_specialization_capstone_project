unzip archive.zip
python3 ./generate_train_test_data.py --image-file-path train-images.idx3-ubyte  --label-file-path train-labels.idx1-ubyte
python3 ./generate_train_test_data.py --image-file-path t10k-images.idx3-ubyte  --label-file-path t10k-labels.idx1-ubyte