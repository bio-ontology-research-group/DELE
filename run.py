import mowl

mowl.init_jvm("8g")

from models.elembeddings_owl2vec_star import ELEmModel
from data_utils.datasets import OWLTwoVecStarDataset
from evaluation_utils import ModelRankBasedEvaluator
import torch as th
import random


random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed_all(0)


dataset = OWLTwoVecStarDataset('data/food_ontology/train_subsumption.owl',
                               'data/food_ontology/valid_subsumption.owl', 
                               'data/food_ontology/test_subsumption.owl')


model = ELEmModel(
    dataset,
    embed_dim=400,
    margin=-0.1,
    epsilon=0.01,
    learning_rate=0.0001,
    epochs=800,
    batch_size=4096 * 8,
    model_filepath='model.pt',
    device="cpu",
    test_gci="gci0",
    eval_property=None,
    path_to_dc=None,
    path_to_test=None,
)


model.train(
    neg_types=["gci2"],
)

with th.no_grad():
    model.load_best_model()
    evaluator = ModelRankBasedEvaluator(
        model, device="cpu", eval_method=model.eval_method, auc_mode="micro", test_gci="gci0", filter_by="train", test="test",
    )

    evaluator.evaluate(show=True)
