#MtRA-Unet 

This repo contains the Official Pytorch implementation of our paper:

    Segmentation of tibiofemoral joint tissues from knee MRI using MtRA-Unet and incorporating shape information: Data from the Osteoarthritis Initiative

    Akshay Daydar*, Alik Pramanick, Arijit Sur, Subramani Kanagaraj

![MtRA-Unet Architecture](./MtRA_Unet_Architecture.png)/

Repository Structure:

MtRA_Unet/
├── train.py
├── test.py
├── README.md
├── train_test_images.csv
└── utils/
    ├── __init__.py   
    ├── config.py
    ├── evaluation_metrics_test.py
    ├── evaluation_metrics_train.py
    ├── loader.py
    ├── loader_test.py
    ├── loss_functions.py
    ├── model.py

Requirements

    Linux
    Python3 3.8.10
    Pytorch 1.13.1
    train and test with A100 GPU

Prepare Dataset:

    1. Download Knee MRI Dataset from OAI repository at https://nda.nih.gov/oai and corresponding masks from OAI-ZIB repository at pubdata.zib.de
    2. Convert the downloaded dicom image files and .mhd mask files to png file format.
    3. Optional: Rename the all files to SubjectID_Imagenumber file format for better coding clearity and results comparison.
    4. Save the preprocessed dataset to "Data2" folder. Within this folder independent "Training" and "Testing" folders need to be made following the image ID provided in "train_test_images.csv" file. 

Training and Testing:

Prepare the dataset and then run the following command for pretrain:

    python3 train.py

For Testing, run

    python3 test.py

Citation:
 If you find this repo useful for your research, please consider citing our paper:

    Daydar, Akshay, et al. "Segmentation of tibiofemoral joint tissues from knee MRI using MtRA-Unet and incorporating shape information: Data from the Osteoarthritis Initiative." arXiv preprint arXiv:2401.12932 (2024).
