<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>Multiclass Movie Genres Classification From Original Movie Overview</b></h1>
<!-- Main -->

## Members
This project is a part of the course **Natural Language Processing** at the University of Information Technology

| No            | Student ID    | Full name            | Email                   |
| ------------- | ------------- | -------------------- | ----------------------- |
| 1             | 23520179      | Phùng Minh Chí       | 23520179@gm.uit.edu.vn  |
| 2             | 23520183      | Nguyễn Hữu Minh Chiến | 23520183@gm.uit.edu.vn  |
| 3             | 23521467      | Lê Ngọc Phương Thảo  | 23521467@gm.uit.edu.vn  |

## Course Information
* **Course** Natural Language Processing 
* **Course code:** CS221
* **Class code:** CS221.P22
* **Semester:** HK2 (2024 - 2025)
* **Instructor**: TS Nguyễn Trọng Chỉnh

## Instruction
1. Clone the repository:
   ```bash
   git clone hhttps://github.com/chisphung/CS221-GenresPrediction-from-Overview
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### Data preprocessing:
To preprocess the dataset, run the following command:
```bash
python tools/preprocess.py 
```
You can also download the preprocessed dataset with the following command:
```bash
python tools/download.py
```
### Model training:
To train the model, run the following command:
```bash
python tools/train.py <pretrained_model_name> <dataset_path>
```
Replace `<pretrained_model_name>` with the name of the pretrained model you want to use (e.g., `bert-base-uncased`) and `<dataset_path>` with the path to your dataset.
### Pretrained model:
To save your time, we are current support 3 pretrained models:
- `bert-base-uncased`
- `distilled-bert-base-uncased`
- `roberta-base`

You can download them from the following links:
- [bert-base-uncased](https://drive.google.com/drive/u/0/folders/1VMI2n7ZvDL6YL5iVGRI3dTr3aJdzDVek)
- [distilled-bert-base-uncased](https://drive.google.com/drive/u/0/folders/1VMI2n7ZvDL6YL5iVGRI3dTr3aJdzDVek)
- [roberta-base](https://drive.google.com/drive/u/0/folders/1VMI2n7ZvDL6YL5iVGRI3dTr3aJdzDVek)

After downloading, you can place them in the `weights` folder.
### Prediction:
To make a single prediction using the trained model, run the following command:
```bash
python -m src.main.py 
```
### Deployment:
To deploy the model using streamlit, run the following command:
```bash
streamlit run src/streamlit.py
```
