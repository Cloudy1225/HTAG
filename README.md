# Multi-Scale Heterogeneous Text-Attributed Graph Datasets From Diverse Domains

- **Multi Scales.** Our HTAG datasets span multiple scales, ranging from small (24K nodes, 104K edges) to large (5.6M nodes, 29.8M edges). Smaller datasets are suitable for testing computationally intensive algorithms, while larger datasets, such as DBLP and Patent, support the development of scalable models that leverage mini-batching and distributed training.
- **Diverse Domains.** Our HTAG datasets include heterogeneous graphs that are representative of a wide range of domains: movie collaboration, community question answering, academic, book publication, and patent application. The broad coverage of domains empowers the development and demonstration of graph foundation models and helps differentiate them from domain-specific approaches.
- **Realistic and Reproducible Evaluation.** We provide an automated evaluation pipeline for HTAGs that streamlines data processing, loading, and model evaluation, ensuring seamless reproducibility. Additionally, we employ time-based data splits for each dataset, which offer a more realistic and meaningful evaluation compared to traditional random splits.
- **Open-source Code for Dataset Construction.** We have released the complete code for constructing our HTAG datasets, allowing researchers to build larger and more complex heterogeneous text-attribute graph datasets. For example, the CroVal dataset construction code can be used to create web-scale community question-answering networks, such as those derived from [StackExchange data dumps](https://archive.org/download/stackexchange). This initiative aims to further advance the field by providing the tools necessary for replicating and extending our datasets for a wide range of applications.



## Datasets

The HTAG datasets and dataset construction code are available at [HTAG · Datasets on Hugging Face](https://huggingface.co/datasets/Cloudy1225/HTAG). Each dataset folder contains a `README.md` file (providing a dataset description and instructions for reproducing dataset construction), a `.pkl` file (containing the heterogeneous graph information), a `.csv` or `.csv.zip` file (with raw text), and a `.ipynb` file (for constructing the dataset).

### Download

```python
from huggingface_hub import snapshot_download

# Download all
snapshot_download(repo_id="Cloudy1225/HTAG", repo_type="dataset", local_dir="./data")

# Or just download heterogeneous graphs and PLM-based node features
snapshot_download(repo_id="Cloudy1225/HTAG", repo_type="dataset", local_dir="./data", allow_patterns="*.pkl")

# Or just download raw texts
snapshot_download(repo_id="Cloudy1225/HTAG", repo_type="dataset", local_dir="./data", allow_patterns=["*.csv", "*.csv.zip"])
```



### Dataset Format

The dataset includes heterogeneous graph edges, raw text, PLM-based features, labels, and years associated with text-attributed nodes. Raw text is provided in `.csv` or `.csv.zip` files, while the remaining data are stored in a dictionary object within a `.pkl` file. For example, by reading the `tmdb.pkl` file, the following dictionary can be obtained:

```
{'movie-actor': (array([   0,    0,    0, ..., 7504, 7504, 7504], dtype=int16),
  array([    0,     1,     2, ..., 11870,  1733, 11794], dtype=int16)),
 'movie-director': (array([   0,    0,    0, ..., 7503, 7503, 7504], dtype=int16),
  array([   0,    1,    2, ..., 3423,  966, 2890], dtype=int16)),
 'movie_labels': array([3, 1, 1, ..., 1, 1, 2], dtype=int8),
 'movie_feats': array([[ 0.00635284,  0.00649689,  0.01250827, ...,  0.06342042,
         -0.01747945,  0.0134356 ],
        [-0.14075027,  0.02825641,  0.02670695, ..., -0.12270895,
          0.08417314,  0.02486392],
        [ 0.00014208, -0.02286632,  0.00615967, ..., -0.03311544,
          0.04735276, -0.07458566],
        ...,
        [ 0.01835816,  0.07484645, -0.08099765, ..., -0.00150019,
          0.01669764,  0.00456595],
        [-0.00821487, -0.10434289,  0.01928608, ..., -0.06343049,
          0.05060194, -0.04229118],
        [-0.06465845,  0.13461556, -0.01640793, ..., -0.06274845,
          0.04002513, -0.00751513]], dtype=float32),
 'movie_years': array([2013, 1995, 1989, ..., 1939, 1941, 1965], dtype=int16)}
```



### Dataset Statistics

|                                                              | # Nodes             | # Edges                    | # Classes | # Splits         |
| ------------------------------------------------------------ | ------------------- | -------------------------- | --------- | ---------------- |
| [TMDB](https://huggingface.co/datasets/Cloudy1225/HTAG/blob/main/tmdb/README.md) | **24,412**          | **104,858**                | 4         | Train: 5,698     |
|                                                              | Movie: 7,505        | Movie-Actor: 86,517        |           | Valid: 711       |
|                                                              | Actor: 13,016       | Movie-Director: 18,341     |           | Test: 1,096      |
|                                                              | Director: 3,891     |                            |           |                  |
| [CroVal](https://huggingface.co/datasets/Cloudy1225/HTAG/blob/main/croval/README.md) | **44386**           | **164,981**                | 6         | Train: 980       |
|                                                              | Question: 34153     | Question-Question: 46,269  |           | Valid: 1,242     |
|                                                              | User: 8898          | Question-User: 34,153      |           | Test: 31,931     |
|                                                              | Tag: 1335           | Question-Tag: 84,559       |           |                  |
| [ArXiv](https://huggingface.co/datasets/Cloudy1225/HTAG/blob/main/arxiv/README.md) | **231,111**         | **2,075,692**              | 40        | Train: 47,084    |
|                                                              | Paper: 81,634       | Paper-Paper: 1,019,624     |           | Valid: 18,170    |
|                                                              | Author: 127,590     | Paper-Author: 300,233      |           | Test: 16,380     |
|                                                              | FoS: 21,887         | Paper-FoS: 755,835         |           |                  |
| [Book](https://huggingface.co/datasets/Cloudy1225/HTAG/blob/main/book/README.md) | **786,257**         | **9,035,291**              | 8         | Train: 330,201   |
|                                                              | Book                | Book-Book: 7,614,902       |           | Valid: 57,220    |
|                                                              | Author              | Book-Author: 825,905       |           | Test: 207,063    |
|                                                              | Publisher           | Book-Publisher: 594,484    |           |                  |
| [DBLP](https://huggingface.co/datasets/Cloudy1225/HTAG/blob/main/dblp/README.md) | **1,989,010**       | **29,830,033**             | 9         | Train: 508,464   |
|                                                              | Paper: 964350       | Paper-Paper: 16,679,526    |           | Valid: 158,891   |
|                                                              | Author: 958961      | Paper-Author: 3,070,343    |           | Test: 296,995    |
|                                                              | FoS: 65699          | Paper-FoS: 10,080,164      |           |                  |
| [Patent](https://huggingface.co/datasets/Cloudy1225/HTAG/blob/main/patent/README.md) | **5,646,139**       | **8,833,738**              | 120       | Train: 1,705,155 |
|                                                              | Patent: 2,762,187   | Patent-Inventor: 6,071,551 |           | Valid: 374,275   |
|                                                              | Inventor: 2,873,311 | Patent-Examiner: 2,762,187 |           | Test: 682,757    |
|                                                              | Examiner: 10,641    |                            |           |                  |



## Experiments

### Datasets Setup

To set up the datasets, download them from [HTAG · Datasets at Hugging Face](https://huggingface.co/datasets/Cloudy1225/HTAG) into the `./data` folder:

```python
from huggingface_hub import snapshot_download

# Download heterogeneous graphs and PLM-based node features
snapshot_download(repo_id="Cloudy1225/HTAG", repo_type="dataset", local_dir="./data", allow_patterns="*.pkl")
```

Subsequently, the `dataloader.py` script will load the `.pkl` file, construct a `dgl.DGLGraph` object, and generate the train/validation/test splits.



### Model Configuration

Model configurations are provided in the `./train.conf.yaml` file.



### Run Experiments

To reproduce the experiment results, execute the following commands:

```python
python MLP.py

python GNN.py --model GCN/SAGE/GAT

python RGNN.py --model RGCN/RSAGE/RGAT/ieHGCN
```



|        | TMDB           |                | CroVal         |                | ArXiv          |                | Book           |                | DBLP           |                | Patent         |                |
| ------ | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
|        | Micro-F1       | Macro-F1       | Micro-F1       | Macro-F1       | Micro-F1       | Macro-F1       | Micro-F1       | Macro-F1       | Micro-F1       | Macro-F1       | Micro-F1       | Macro-F1       |
| MLP    | 72.32±0.50     | 71.93±0.50     | 85.78±0.23     | 83.26±0.24     | 78.97±0.20     | 43.51±1.03     | 75.07±0.09     | 66.63±0.16     | 69.91±0.02     | 65.46±0.05     | 69.64±0.03     | 52.69±0.16     |
| GCN    | 77.39±0.18     | 78.24±0.19     | 82.83±0.32     | 79.42±0.47     | 81.95±0.20     | 49.83±0.92     | 80.72±0.08     | 73.91±0.15     | 73.91±0.15     | 70.26±0.20     | 73.64±0.12     | 56.61±0.39     |
| SAGE   | 79.29±0.57     | 80.06±0.59     | 85.26±0.37     | 82.84±0.45     | 83.89±0.14     | 52.18±0.54     | 81.43±0.07     | 74.83±0.11     | 75.62±0.07     | 72.43±0.09     | 75.58±0.14     | 59.54±0.79     |
| GAT    | 79.69±0.27     | 80.40±0.23     | 83.37±0.36     | 80.56±0.36     | 83.84±0.41     | 50.24±1.82     | 80.96±0.07     | 74.19±0.16     | 75.62±0.20     | 72.49±0.28     | 74.74±0.12     | 56.30±0.49     |
| RGCN   | 81.57±0.74     | 82.12±0.67     | 87.10±0.26     | 84.47±0.38     | 84.80±0.30     | 54.01±0.79     | 82.22±0.11     | 76.57±0.14     | 77.40±0.09     | 74.61±0.12     | 76.97±0.08     | 60.67±0.12     |
| RSAGE  | **82.24±0.32** | **82.76±0.32** | **87.44±0.26** | **85.00±0.35** | **84.85±0.22** | 54.24±1.47     | 82.35±0.02     | 76.70±0.10     | 77.74±0.13     | 74.95±0.21     | 76.98±0.07     | 61.01±0.39     |
| RGAT   | 81.97±0.16     | 82.40±0.18     | 87.43±0.18     | 84.93±0.22     | 84.69±0.25     | **54.62±1.49** | **82.45±0.04** | **76.75±0.06** | **77.81±0.29** | **74.98±0.33** | **77.22±0.08** | **61.36±0.18** |
| ieHGCN | 81.75±0.47     | 82.19±0.51     | 87.15±0.24     | 84.58±0.31     | 84.08±0.31     | 53.49±1.87     | 81.92±0.43     | 76.10±0.70     | 77.47±1.12     | 74.60±1.32     | 76.90±0.13     | 60.39±0.39     |