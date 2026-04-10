# DeepLSH: Deep Locality-Sensitive Hash Learning for Fast and Efficient Near-Duplicate Crash Report Detection

## Overview
Automatic crash bucketing is a critical step in the software development process to efficiently analyze and triage bug reports. In this work, we aim at detecting for a crash report its candidate near-duplicates (i.e., similar crashes that are likely to be induced by the same software bug) in a large database of historical crashes and given any similarity measure dedicated to compare between stack traces. To this end, we propose **DeepLSH** a deep Siamese hash coding neural network based on Locality-Sensitive Hashing (LSH) property in order to provide binary hash codes aiming to locate the most similar stack traces into hash buckets. **DeepLSH** have been conducted on a large stack trace dataset and performed on state-of-the-art similarity measures proposed to tackle the crash deduplication problem:
- Jaccard coefficient [[Ref](https://en.wikipedia.org/wiki/Jaccard_index)]
- Cosine similarity [[Ref](https://en.wikipedia.org/wiki/Sine_and_cosine)]
- Lucene TF-IDF [[Ref](https://lucene.apache.org/core/7_6_0/core/org/apache/lucene/search/similarities/TFIDFSimilarity.html)]
- Edit distance [[Ref](https://en.wikipedia.org/wiki/Edit_distance)]
- Brodie et al. [[Paper](https://www.cs.drexel.edu/~spiros/teaching/CS576/papers/Brodie_ICAC05.pdf)]
- PDM-Rebucket [[Paper](https://www.researchgate.net/publication/254041628_ReBucket_A_method_for_clustering_duplicate_crash_reports_based_on_call_stack_similarity)]
- DURFEX [[Paper](https://users.encs.concordia.ca/~abdelw/papers/QRS17-Durfex.pdf)]
- Lerch and Mezini [[Paper](https://files.inria.fr/sachaproject/htdocs//lerch2013.pdf)]
- Moroo et al. [[Paper](http://ksiresearch.org/seke/seke17paper/seke17paper_135.pdf)]
- TraceSIM [[Paper](https://arxiv.org/pdf/2009.12590.pdf)]

## Contributions

Our contribution is three-fold. 
- Aiming to overcome the problem of deriving LSH functions for stack-trace similarity measures, we propose a generic approach dubbed DeepLSH that learns and provides a family of binary hash functions that perfectly approximate the locality-sensitive property to retrieve efficiently and rapidly near-duplicate stack traces. 

![lsh](code/Images/lshPhases.png)

- Technically, we design a deep Siamese neural network architecture to perform end-to-end hashing with an original objective loss function based on the locality-sensitive property preserving with appropriate regularizations to cope with the binarization problem of optimizing non-smooth loss functions. 
- We demonstrate through our experimental study the effectiveness and scalability of DeepLSH to yield near-duplicate crash reports under a dozen of similarity metrics. We successfully compare to standard LSH techniques (MinHash and SimHash), and the most relevant deep hashing baselineon a large real-world dataset that we make available.

![contrib](code/Images/Images-paper/DeepLSH%20model.png)

## How to use this code? 推荐使用的是python3.9

1. Clone this repository 
2. Install the required python packages: ```pip install -r ./code/requirements.txt ```
3. Run without Jupyter (recommended for local CLI usage):
    
    3.1. Create and activate a virtual environment (Windows PowerShell example):
    ```
    py -3.9 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r .\code\requirements.txt
    ```

    3.2. List available similarity measures (from `data/similarity-measures-pairs.csv`):
    ```
    python .\code\run.py list
    ```

    3.3. Lightweight (lite) run (fast): query one similarity value from the precomputed pairs file.
    This is the fastest way to "run the project" locally and choose a measure.
    ```
    python .\code\run.py lite --measure TraceSim --index-a 0 --index-b 10
    python .\code\run.py lite --measure Jaccard --index-a 0 --index-b 10
    python .\code\run.py lite --measure Brodie --index-a 0 --index-b 10
    python .\code\run.py lite --measure DURFEX --index-a 0 --index-b 10
    python .\code\run.py lite --measure TfIdf --index-a 0 --index-b 10
    ```
    Note: the provided `similarity-measures-pairs.csv` corresponds to 1000 stacks in this repo.

    3.4. Full run (DeepLSH training + LSH hash tables): train DeepLSH for a selected measure and build hash tables.
    ```
    python .\code\run.py deeplsh --measure TraceSim
    python .\code\run.py deeplsh --measure Jaccard
    python .\code\run.py deeplsh --measure Cosine
    python .\code\run.py deeplsh --measure TfIdf
    python .\code\run.py deeplsh --measure Levensh
    python .\code\run.py deeplsh --measure PDM
    python .\code\run.py deeplsh --measure Brodie
    python .\code\run.py deeplsh --measure DURFEX
    python .\code\run.py deeplsh --measure Lerch
    python .\code\run.py deeplsh --measure Moroo
    ```
    Outputs:
    - Models are saved to `code/Models/` as `model-deep-lsh-<measure>.model`
    - Hash tables are saved to `code/Hash-Tables/` as `hash_tables_deeplsh_<measure>.pkl`

    3.5. Faster smoke test for DeepLSH (recommended first run):
    ```
    python .\code\run.py deeplsh --measure TraceSim --n 200 --epochs 1 --batch-size 128
    ```

 4. (Optional) Run the notebooks in `code/notebooks/` if you prefer the original experimental setup.
