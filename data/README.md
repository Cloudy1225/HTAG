---
license: mit
task_categories:
- graph-ml
- text-classification
language:
- en
size_categories:
- 1M<n<10M
---



# Multi-Scale Heterogeneous Text-Attributed Graph Datasets From Diverse Domains

- **Multi Scales.** Our HTAG datasets span multiple scales, ranging from small (24K nodes, 104K edges) to large (5.6M nodes, 29.8M edges). Smaller datasets are suitable for testing computationally intensive algorithms, while larger datasets, such as DBLP and Patent, support the development of scalable models that leverage mini-batching and distributed training.
- **Diverse Domains.** Our HTAG datasets include heterogeneous graphs that are representative of a wide range of domains: movie collaboration, community question answering, academic, book publication, and patent application. The broad coverage of domains empowers the development and demonstration of graph foundation models and helps differentiate them from domain-specific approaches. 
- **Realistic and Reproducible Evaluation.** We provide an automated evaluation pipeline for HTAGs that streamlines data processing, loading, and model evaluation, ensuring seamless reproducibility. Additionally, we employ time-based data splits for each dataset, which offer a more realistic and meaningful evaluation compared to traditional random splits.
- **Open-source Code for Dataset Construction.** We have released the complete code for constructing our HTAG datasets, allowing researchers to build larger and more complex heterogeneous text-attribute graph datasets. For example, the CroVal dataset construction code can be used to create web-scale community question-answering networks, such as those derived from [StackExchange data dumps](https://archive.org/download/stackexchange). This initiative aims to further advance the field by providing the tools necessary for replicating and extending our datasets for a wide range of applications.



## Download

```python
from huggingface_hub import snapshot_download

# Download all
snapshot_download(repo_id="Cloudy1225/HTAG", repo_type="dataset", local_dir="./data")

# Or just download heterogeneous graphs and PLM-based node features
snapshot_download(repo_id="Cloudy1225/HTAG", repo_type="dataset", local_dir="./data", allow_patterns="*.pkl")

# Or just download raw texts
snapshot_download(repo_id="Cloudy1225/HTAG", repo_type="dataset", local_dir="./data", allow_patterns=["*.csv", "*.csv.zip"])
```



## Dataset Format

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



## Dataset Statistics

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



## Dataset Construction

The code for dataset construction can be found in each `graph_builder.ipynb` file. Please see `README.md` in each subfolder for more details .