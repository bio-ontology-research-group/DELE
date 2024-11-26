This is the code for the paper [DELE: Deductive $\mathcal{EL++}$ Embeddings for Knowledge Base Completion](https://arxiv.org/abs/2411.01574).

## Repository Overview

* *run.py* is an example of how to train and evaluate a model
* *evaluation_utils.py*: rank-based evaluator
* *losses/elembeddings_losses.py*: GCI loss functions for ELEmbeddings model
* *losses/elbe_losses.py*: GCI loss functions for ELBE model
* *losses/box2el_losses.py*: GCI loss functions for Box2EL model
* *data_utils/datasets.py* dataset classes
* *data_utils/dataloader_go.py*: ontology dataloader for GO & STRING data
* *data_utils/dataloader_owl2vec_star.py*: ontology dataloader for OWL2Vec* data
* *data_utils/deductive_closure.py*: deductive closure computation
* *models/elembeddings_go.py*: ELEmbeddings model for PPI and function prediction
* *models/elembeddings_owl2vec_star.py*: ELEmbeddings model for FoodOn data
* *models/elbe_go.py*: ELBE model for PPI and function prediction
* *models/elbe_owl2vec_star.py*: ELBE model for FoodOn data
* *models/box2el_go.py*: Box2EL model for PPI and function prediction
* *models/box2el_owl2vec_star.py*: Box2EL model for FoodOn data
* *data/food_ontology*: contains train/validation/test sets for FoodOn
* *data/go_string/\*_yeast.owl*: train/validation/test sets for PPI prediction
* *data/go_string/\*_yeast_hf.owl*: train/validation/test sets for protein function prediction

### Dependencies

* Python 3.9
* Anaconda

### Set up environment

```
git clone https://github.com/bio-ontology-research-group/DELE.git
cd DELE
conda env create -f environment.yml
conda activate dele_env
```