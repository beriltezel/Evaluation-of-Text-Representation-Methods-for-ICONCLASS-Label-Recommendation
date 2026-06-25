# Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation
Bachelor thesis repository of Beril Tezel

This repository contains the final versions of the search methods and the first working version of the evaluation pipeline for comparing different text representation methods for Iconclass label recommendation. Older versions of the code are kept separately in the `older_versions` folder, so the main files in the repository are easier to follow.

## Final files

`build_iconclass_hierarchy_final.py`: this file builds the Iconclass hierarchy database used in the evaluation step. The database contains one main SQL table with the columns `notation`, `lang`, `label`, `parent`, and `depth`. Here, `notation` is the Iconclass code, `lang` is the language of the label, `label` is the textual Iconclass label, `parent` is the parent notation in the hierarchy, and `depth` is the level of the notation in the hierarchy. The `parent` and `depth` columns are especially important for calculating Wu-Palmer similarity later.

The script also uses `used_notation_keys.txt`. In the final setup, normal Iconclass notations are kept, but notations with additions such as `(+...)` are only kept if they are listed in `used_notation_keys.txt`. This keeps the searchable Iconclass set consistent across the models.

`bm25_final.py`: this file contains the final BM25 search version used for evaluation. It reads the same Iconclass candidate set described above, loads the ground truth from the CSV file, runs BM25 for every query, and saves the ranked results.

`sbert_build_embeddings_final.py`: this file builds the SBERT embeddings from the filtered Iconclass hierarchy database. The embeddings and metadata are saved so that the search file can use them later without rebuilding them every time.

`sbert_search_final.py`: this file contains the final SBERT search version used for evaluation. It loads the saved SBERT embeddings, reads the ground-truth CSV file, runs the search for every query, and saves the ranked results.

`gemma_final.ipynb`: this notebook contains the final Gemma embedding search version used for evaluation. It builds the Gemma embedding table, reads the ground-truth CSV file, runs the search for every query, and saves the ranked results.

`siglip_final.ipynb`: this notebook contains the final SigLIP search version used for evaluation. It builds the SigLIP embedding table, reads the ground-truth CSV file, runs the search for every query, and saves the ranked results.

All final model files read the ground truth from a CSV file converted from the Excel ground-truth file. The CSV uses `;` as the separator. The first column contains the query text, and the following columns contain the relevant Iconclass notations.

Each model saves its output in a JSONL file named `model_results.jsonl`. Each line contains one query result, including the model name, query, ground-truth codes, predicted codes, ranks, labels, and scores.

`evaluation_final.py`: this file reads the saved model outputs from `model_results.jsonl` and evaluates the results. It also uses the Iconclass hierarchy database for the hierarchy-based metric.

The current evaluation metrics are:

- precision
- recall
- F1-score
- R-precision
- mean average precision
- Wu-Palmer similarity

For Wu-Palmer, the script calculates the similarity between predicted and ground-truth Iconclass codes using their position in the hierarchy. It saves separate values for ground-truth-to-prediction coverage and prediction-to-ground-truth closeness. The Wu-Palmer mean is the average of these two values.

The evaluation results are saved in `evaluation_results.jsonl`. The script also creates basic graphs for the current results. These graphs are meant as a first working version and can still be improved later.

`test_json_reader.py`: this is a small helper file for reading and checking the JSONL result files. It is mainly used to quickly inspect whether the saved model and evaluation outputs look correct.
