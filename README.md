## COS30049 Assignment 2
Computing Technology Innovation Project<br>
AI-based Project for Cybersecurity (AI4Cyber)<br>
Swinburne University of Technology<br>
Session 7 Group 4

**Group Members**<br>
Angelina The 103162372<br>
Clarice Victoria Zi Wang Shim 102788227<br>
Rye Mohammed Fahim 104557267 

## Folder Structure<br>
**Raw Data** - Contains pre-cleaned and cleaned data of datasets used in this project<br>
**Data Processing** - Contains scripts to split, join, and clean data<br>
**Final Data** - Contains processed data (for testing, training, validation)<br>
**Models** - Contains Classification, Clustering, and CNN Models

## How to run our project:
### Prerequisites
- Python **3.10+** (3.12 tested)
- Git
- Recommended: virtual environment tool (`venv` or `conda`)
- Make sure you have all data and embeddings:
```
    Final Data/train.csv
    Final Data/test.csv
    Final Data/val.csv
    Models/CNN/embeddings_cnn/emb_train.npy
    Models/CNN/embeddings_cnn/emb_test.npy
```

### 1. Clone Repository and Install Dependencies
Clone repository and enter SUBMISSION directory
```
git clone https://github.com/ryefahim/COS30049.git
cd COS30049/SUBMISSION
```

Pip install requirements
```
pip install -r requirements.txt
```

### 2. Running Individual Models

#### Running Classification Model
*This is Jupyter notebook file. Please ensure you have Jupyter Notebook setup on your preferred IDE.*

Simply open the Models/Classification folder, select the ```NBmodel.ipynb```, and click "Run all". 

Optionally, 
1. Launch Jupyter Notebook from Terminal
2. Open Models/Classification/NBmodel.ipynb in the browser and run all cells
3. Results would be stored under the same /Models/Classification/ Folder. 


#### Running Clustering Model
Simply open the folder, select the file ```clustering_without_embeddings.py``` or ```clustering_with_embeddings.py```, and click "Run". 

Optionally, you can run these Python files from terminal:

**Run clustering without embeddings**
```
python Models/Clustering/clustering_without_embeddings.py
```
Results stored under /Models/Clustering/results/without_embeddings

**Run clustering with embeddings**
```
python Models/Clustering/clustering_with_embeddings.py
```
Results stored under /Models/Clustering/results/with_embeddings


#### Running CNN Model
*This is Jupyter notebook file. Please ensure you have Jupyter Notebook setup on your preferred IDE.*

Simply open the Models/CNN folder, select the ```train_cnn.ipynb```, and click "Run all". <br>
Simply open the Models/CNN folder, select the ```eval_and_export_cnn.ipynb```, and click "Run all". 

Optionally, 
1. Launch Jupyter Notebook from Terminal
2. Open Models/Classification/train_cnn.ipynb in the browser and run all cells
3. Open Models/Classification/eval_and_export_cnn.ipynb in the browser and run all cells
4. Results would be stored under /Models/CNN/ folder

