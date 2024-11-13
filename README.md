#MtRA-Unet 

This repo contains the Official Pytorch implementation of our paper:

    Segmentation of tibiofemoral joint tissues from knee MRI using MtRA-Unet and incorporating shape information: Data from the Osteoarthritis Initiative

    Akshay Daydar*, Alik Pramanick, Arijit Sur, Subramani Kanagaraj

Requirements

    Linux
    Python3 3.8.10
    Pytorch 1.13.1
    train and test with A100 GPU

Prepare Dataset:

    Download Knee MRI Dataset from OAI repository at https://nda.nih.gov/oai and corresponding masks from OAI-ZIB repository at pubdata.zib.de.

Training and Testing:

Prepare the dataset and then run the following command for pretrain:

    python3 train.py

For Testing, run

    python3 test.py

Citation:
 If you find this repo useful for your research, please consider citing our paper:

    Daydar, Akshay, et al. "Segmentation of tibiofemoral joint tissues from knee MRI using MtRA-Unet and incorporating shape information: Data from the Osteoarthritis Initiative." arXiv preprint arXiv:2401.12932 (2024).
