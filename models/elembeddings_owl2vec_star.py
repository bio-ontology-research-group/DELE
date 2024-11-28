import mowl

mowl.init_jvm("8g")

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELModule
from mowl.projection.factory import projector_factory
from losses.elembeddings_losses import *
from data_utils.dataloader_owl2vec_star import OntologyDataLoader
import torch as th
from torch import nn
from torch.nn.functional import relu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import numpy as np
from mowl.datasets import PathDataset


class ELEmModule(ELModule):
    def __init__(
        self,
        nb_ont_classes,
        nb_rels,
        embed_dim=50,
        margin=0.1,
        epsilon=0.001,
        test_gci="gci2",
    ):
        """
        ELEmbeddings module

        :param nb_ont_classes: total number of classes
        :type nb_ont_classes: int
        :param nb_rels: total number of relations
        :type nb_rels: int
        :param embed_dim: embedding dimension
        :type embed_dim: int
        :param margin: margin parameter \gamma
        :type margin: float/int
        :param epsilon: $\varepsilon$ parameter for negative loss computation
        :type epsilon: float
        :param test_gci: GCI test type (`gci0` or `gci2`)
        :type test_gci: str
        """
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.embed_dim = embed_dim
        self.test_gci = test_gci

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(self.class_embed.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_embed.weight.data /= weight_data_normalized

        self.class_rad = nn.Embedding(self.nb_ont_classes, 1)
        nn.init.uniform_(self.class_rad.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(
            self.class_rad.weight.data, axis=1
        ).reshape(-1, 1)
        self.class_rad.weight.data /= weight_data_normalized

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(
            self.rel_embed.weight.data, axis=1
        ).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data_normalized

        self.margin = margin
        self.epsilon = epsilon

    def class_reg(self, x):
        """
        Regularization function

        :param x: point to regularize
        :type x: torch.Tensor(torch.float64)
        :return res: regularized point
        :type res: torch.Tensor(torch.float64)
        """
        if self.reg_norm is None:
            res = th.zeros(x.size()[0], 1)
        else:
            res = th.abs(th.linalg.norm(x, axis=1) - 1)
            res = th.reshape(res, [-1, 1])
        return res

    def gci0_loss(self, data, neg=False):
        """
        Compute GCI0 (`C \sqsubseteq D`) loss

        :param data: GCI0 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        return gci0_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            neg=neg,
        )

    def gci0_bot_loss(self, data, neg=False):
        """
        Compute GCI0_BOT (`C \sqsubseteq \bot`) loss

        :param data: GCI0_BOT data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        return gci0_bot_loss(data, self.class_rad, self.epsilon, neg=neg)

    def gci1_loss(self, data, neg=False):
        """
        Compute GCI1 (`C \sqcap D \sqsubseteq E`) loss

        :param data: GCI1 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        return gci1_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            neg=neg,
        )

    def gci1_bot_loss(self, data, neg=False):
        """
        Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) loss

        :param data: GCI1_BOT data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        return gci1_bot_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            neg=neg,
        )

    def gci2_loss(self, data, neg=False, idxs_for_negs=None):
        """
        Compute GCI2 (`C \sqsubseteq \exists R.D`) loss

        :param data: GCI2 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        return gci2_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.rel_embed,
            self.class_reg,
            self.margin,
            neg=neg,
        )

    def gci3_loss(self, data, neg=False):
        """
        Compute GCI3 (`\exists R.C \sqsubseteq D`) loss

        :param data: GCI3 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        return gci3_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.rel_embed,
            self.class_reg,
            self.margin,
            neg=neg,
        )

    def gci3_bot_loss(self, data, neg=False):
        """
        Compute GCI3_BOT (`\exists R.C \sqsubseteq \bot`) loss

        :param data: GCI3_BOT data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        return gci3_bot_loss(data, self.class_rad, self.epsilon, neg=neg)

    def eval_method(self, data):
        """
        Compute evaluation score (for GCI2 `C \sqsubseteq \exists R.D` or GCI0 `C \sqsubseteq D` axioms)

        :param data: evaluation data
        :type data: torch.Tensor(torch.int64)
        :return: evaluation score value for each data sample
        :return type: torch.Tensor(torch.float64)
        """
        if self.test_gci == "gci0":
            return gci0_loss(
                data,
                self.class_embed,
                self.class_rad,
                None,
                self.margin,
                neg=False,
            )
        elif self.test_gci == "gci2":
            return gci2_loss(
                data,
                self.class_embed,
                self.class_rad,
                self.rel_embed,
                None,
                self.margin,
                neg=False,
            )


class ELEmbeddings(EmbeddingELModel):
    def __init__(
        self,
        dataset,
        embed_dim=50,
        margin=0,
        epsilon=0.001,
        learning_rate=0.001,
        epochs=1000,
        batch_size=4096 * 8,
        model_filepath=None,
        device="cpu",
        test_gci="gci2",
        eval_property=None,
        path_to_dc=None,
        path_to_test=None,
    ):
        """
        ELEmbeddings model

        :param dataset: dataset to use
        :type dataset: data_utils.data.PPIYeastDataset/data_utils.data.AFPYeastDataset
        :param embed_dim: embedding dimension
        :type embed_dim: int
        :param margin: margin parameter \gamma
        :type margin: float/int
        :param epsilon: $\varepsilon$ parameter for negative loss computation
        :type epsilon: float
        :param learning_rate: learning rate
        :type learning_rate: float
        :param epochs: number of training epochs 
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param model_filepath: path to model checkpoint
        :type model_filepath: str
        :param device: device to use, e.g., `cpu`, `cuda`
        :type device: str
        :param eval_property: evaluation property
        :type eval_property: str
        :param path_to_dc: path to the deductive closure, need to provide if metrics are filtered with respect to the deductive closure
        :type path_to_dc: str
        :param path_to_test: path to the test (if differs from the default test set)
        :type path_to_test: str
        """
        super().__init__(
            dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath
        )

        self.embed_dim = embed_dim
        self.margin = margin
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.test_gci = test_gci
        self.eval_property = eval_property
        if path_to_dc is not None:
            self.dc = PathDataset(path_to_dc).ontology
        else:
            self.dc = None
        if path_to_test is not None:
            self.new_test = PathDataset(path_to_test).ontology
        else:
            self.new_test = None
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_model()

    def init_model(self):
        """
        Load ELEmbeddings module
        """
        self.module = ELEmModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
            epsilon=self.epsilon,
            test_gci=self.test_gci,
        ).to(self.device)
        self.eval_method = self.module.eval_method

    def load_eval_data(self):
        """
        Load evaluation data
        """
        if self._loaded_eval:
            return

        eval_classes = self.dataset.evaluation_classes.as_str

        self._head_entities = set(list(eval_classes)[:])
        self._tail_entities = set(list(eval_classes)[:])

        if self.test_gci == "gci0":
            eval_projector = projector_factory("taxonomy")
        elif self.test_gci == "gci2":
            eval_projector = projector_factory(
                "taxonomy_rels", taxonomy=False, relations=[self.eval_property]
            )
        else:
            raise NotImplementedError

        if self.dc is not None:
            self._dc_set = eval_projector.project(self.dc)
        if self.new_test is not None:
            self._new_test_set = eval_projector.project(self.new_test)
        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True

    def get_embeddings(self):
        """
        Get embeddings of relations and classes from the model checkpoint

        :return ent_embeds: dictionary class_name: its embedding
        :type ent_embeds: dict(str, numpy.array(numpy.float64))
        :return rel_embeds: dictionary relation_name: its embedding
        :type rel_embeds: dict(str, numpy.array(numpy.float64))
        """
        self.init_model()

        print("Load the best model", self.model_filepath)
        self.load_best_model()

        ent_embeds = {
            k: v
            for k, v in zip(
                self.class_index_dict.keys(),
                self.module.class_embed.weight.cpu().detach().numpy(),
            )
        }
        rel_embeds = {
            k: v
            for k, v in zip(
                self.object_property_index_dict.keys(),
                self.module.rel_embed.weight.cpu().detach().numpy(),
            )
        }
        return ent_embeds, rel_embeds

    def load_best_model(self):
        """
        Load the model from the checkpoint
        """
        self.init_model()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

    @property
    def new_test_set(self):
        """
        Get a set of triples that are true positives from the new test dataset
        """
        self.load_eval_data()
        return self._new_test_set

    @property
    def dc_set(self):
        """
        Get a set of triples that are true positives from the deductive closure dataset
        """
        self.load_eval_data()
        return self._dc_set

    @property
    def training_set(self):
        """
        Get a set of triples that are true positives from the train dataset
        """
        self.load_eval_data()
        return self._training_set

    @property
    def testing_set(self):
        """
        Get a set of triples that are true positives from the test dataset
        """
        self.load_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        """
        Get a set of head entities `h` from triples `(h, r, t)`
        """
        self.load_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        """
        Get a set of tail entities `t` from triples `(h, r, t)`
        """
        self.load_eval_data()
        return self._tail_entities

    def train(self):
        raise NotImplementedError


class ELEmModel(ELEmbeddings):
    """
    Final ELEmbeddings model for OWL2Vec* data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(
        self,
        patience=10,
        epochs_no_improve=20,
        path_to_dc=None,
        neg_types=["gci2"],
        random_neg_fraction=1.0,
    ):
        """
        Model training

        :param patience: patience parameter for the scheduler
        :type patience: int
        :param epochs_no_improve: for how many epochs validation loss doesn't improve
        :type epochs_no_improve: int
        :param path_to_dc: absolute path to deductive closure ontology, need to provide if filtered negative sampling strategy is chosen or random_neg_fraction is less than 1
        :type path_to_dc: str
        :param prefix: protein prefix, need to provide for separating GO functions and proteins
        :type prefix: str
        :param neg_types: abbreviations of GCIs to use for negative sampling (`gci0`, `gci1`, `gci2`, `gci3`, `gci0_bot`, `gci1_bot`, `gci3_bot`)
        :type neg_types: list(str)
        :param random_neg_fraction: the fraction of random negatives (the rest negatives are sampled from the deductive closure), should be between 0 and 1
        :type random_neg_fraction: float/int
        """
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        no_improve = 0
        best_loss = float("inf")

        if path_to_dc is not None:
            train_dataloader = OntologyDataLoader(
                self.training_datasets["gci0"][:],
                self.training_datasets["gci1"][:],
                self.training_datasets["gci2"][:],
                self.training_datasets["gci3"][:],
                self.training_datasets["gci0_bot"][:],
                self.training_datasets["gci1_bot"][:],
                self.training_datasets["gci3_bot"][:],
                self.batch_size,
                self.device,
                self.dataset.evaluation_classes.as_str,
                negative_mode="filtered",
                path_to_dc=path_to_dc,
                class_index_dict=self.class_index_dict,
                object_property_index_dict=self.object_property_index_dict,
                random_neg_fraction=random_neg_fraction,
            )
        else:
            train_dataloader = OntologyDataLoader(
                self.training_datasets["gci0"][:],
                self.training_datasets["gci1"][:],
                self.training_datasets["gci2"][:],
                self.training_datasets["gci3"][:],
                self.training_datasets["gci0_bot"][:],
                self.training_datasets["gci1_bot"][:],
                self.training_datasets["gci3_bot"][:],
                self.batch_size,
                self.device,
                self.dataset.evaluation_classes.as_str,
                negative_mode="random",
                class_index_dict=self.class_index_dict,
                object_property_index_dict=self.object_property_index_dict,
                random_neg_fraction=random_neg_fraction,
            )
        num_steps = train_dataloader.num_steps

        for epoch in trange(self.epochs):
            self.module.train()

            train_loss = 0

            for batch in train_dataloader:
                cur_loss = 0
                (
                    gci0,
                    gci0_neg,
                    gci1,
                    gci1_neg,
                    gci2,
                    gci2_neg,
                    gci3,
                    gci3_neg,
                    gci0_bot,
                    gci0_bot_neg,
                    gci1_bot,
                    gci1_bot_neg,
                    gci3_bot,
                    gci3_bot_neg,
                ) = batch
                if len(gci0) > 0:
                    pos_loss = self.module(gci0, "gci0")
                    if "gci0" not in neg_types:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci0_neg, "gci0", neg=True)
                        )
                    cur_loss += l
                if len(gci1) > 0:
                    pos_loss = self.module(gci1, "gci1")
                    if "gci1" not in neg_types:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci1_neg, "gci1", neg=True)
                        )
                    cur_loss += l
                if len(gci2) > 0:
                    pos_loss = self.module(gci2, "gci2")
                    if "gci2" not in neg_types:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(th.square(pos_loss)) + th.mean(
                            th.square(self.module(gci2_neg, "gci2", neg=True))
                        )
                    cur_loss += l
                if len(gci3) > 0:
                    pos_loss = self.module(gci3, "gci3")
                    if "gci3" not in neg_types:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci3_neg, "gci3", neg=True)
                        )
                    cur_loss += l
                if len(gci0_bot) > 0:
                    pos_loss = self.module(gci0_bot, "gci0_bot")
                    if "gci0_bot" not in neg_types:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci0_bot_neg, "gci0_bot", neg=True)
                        )
                    cur_loss += l
                if len(gci1_bot) > 0:
                    pos_loss = self.module(gci1_bot, "gci1_bot")
                    if "gci1_bot" not in neg_types:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci1_bot_neg, "gci1_bot", neg=True)
                        )
                    cur_loss += l
                if len(gci3_bot) > 0:
                    pos_loss = self.module(gci3_bot, "gci3_bot")
                    if "gci3_bot" not in neg_types:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci3_bot_neg, "gci3_bot", neg=True)
                        )
                    cur_loss += l
                train_loss += cur_loss.detach().item()
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

            train_loss /= num_steps

            loss = 0
            with th.no_grad():
                self.module.eval()
                valid_loss = 0
                if self.test_gci == "gci0":
                    gci0_data = self.validation_datasets["gci0"][:]
                    loss = th.mean(self.module(gci0_data, "gci0"))
                elif self.test_gci == "gci2":
                    gci2_data = self.validation_datasets["gci2"][:]
                    loss = th.mean(self.module(gci2_data, "gci2"))
                valid_loss += loss.detach().item()
                scheduler.step(valid_loss)

            if best_loss > valid_loss:
                best_loss = valid_loss
                th.save(self.module.state_dict(), self.model_filepath)
                print(f"Best loss: {best_loss}, epoch: {epoch}")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve == epochs_no_improve:
                print(f"Stopped at epoch {epoch}")
                break

    def eval_method(self, data):
        """
        Evaluation method

        :param data: data for evaluation
        :type data: torch.Tensor(torch.int64)
        """
        return self.module.eval_method(data)
