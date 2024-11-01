import mowl

mowl.init_jvm("8g", "1g", 8)

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELModule
from mowl.projection.factory import projector_factory
from losses.box2el_losses import *
from data_utils.dataloader_owl2vec_star import OntologyDataLoader
import torch as th
from torch import nn
from torch.nn.functional import relu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import numpy as np
from mowl.datasets import PathDataset


class BoxSquaredELModule(ELModule):
    """
    Implementation of Box :math:`^2` EL from [jackermeier2023]_.
    """

    def __init__(
        self,
        nb_ont_classes,
        nb_rels,
        embed_dim=50,
        gamma=0,
        delta=2,
        epsilon=0.001,
        reg_factor=0.05,
        test_gci="gci2",
    ):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.reg_factor = reg_factor
        self.embed_dim = embed_dim
        self.test_gci = test_gci

        self.class_center = self.init_embeddings(nb_ont_classes, embed_dim)
        self.class_offset = self.init_embeddings(self.nb_ont_classes, embed_dim)

        self.head_center = self.init_embeddings(nb_rels, embed_dim)
        self.head_offset = self.init_embeddings(nb_rels, embed_dim)
        self.tail_center = self.init_embeddings(nb_rels, embed_dim)
        self.tail_offset = self.init_embeddings(nb_rels, embed_dim)

        self.bump = self.init_embeddings(nb_ont_classes, embed_dim)

    def init_embeddings(self, num_entities, embed_dim, min=-1, max=1):
        embeddings = nn.Embedding(num_entities, embed_dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        embeddings.weight.data /= th.linalg.norm(
            embeddings.weight.data, axis=1
        ).reshape(-1, 1)
        return embeddings

    def gci0_loss(self, data, neg=False):
        return gci0_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
            self.embed_dim,
            self.epsilon,
            neg=neg,
        )

    def gci0_bot_loss(self, data, neg=False):
        return gci0_bot_loss(
            data,
            self.embed_dim,
            self.class_offset,
            self.epsilon,
            neg=neg,
        )

    def gci1_loss(self, data, neg=False):
        return gci1_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
            self.embed_dim,
            neg=neg,
        )

    def gci1_bot_loss(self, data, neg=False):
        return gci1_bot_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
            self.embed_dim,
            self.epsilon,
            neg=neg,
        )

    def gci2_loss(self, data, neg=False):
        return gci2_loss(
            data,
            self.class_center,
            self.class_offset,
            self.head_center,
            self.head_offset,
            self.tail_center,
            self.tail_offset,
            self.bump,
            self.gamma,
            self.delta,
            self.embed_dim,
            neg=neg,
        )

    def gci3_loss(self, data, neg=False):
        return gci3_loss(
            data,
            self.class_center,
            self.class_offset,
            self.head_center,
            self.head_offset,
            self.bump,
            self.gamma,
            self.delta,
            self.embed_dim,
            neg=neg,
        )

    def gci3_bot_loss(self, data, neg=False):
        return gci3_bot_loss(
            data,
            self.embed_dim,
            self.class_offset,
            self.epsilon,
            neg=neg,
        )

    def regularization_loss(self):
        return reg_loss(self.bump, self.reg_factor)

    def eval_method(self, data):
        if self.test_gci == "gci0":
            return gci0_loss(
                data,
                self.class_center,
                self.class_offset,
                self.gamma,
                self.embed_dim,
                self.epsilon,
                neg=False,
            )
        elif self.test_gci == "gci2":
            return gci2_loss(
                data,
                self.class_center,
                self.class_offset,
                self.head_center,
                self.head_offset,
                self.tail_center,
                self.tail_offset,
                self.bump,
                self.gamma,
                self.delta,
                self.embed_dim,
                neg=False,
            )


class BoxSquaredEL(EmbeddingELModel):
    def __init__(
        self,
        dataset,
        embed_dim=50,
        gamma=0,
        delta=2,
        epsilon=0.001,
        reg_factor=0.05,
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
        super().__init__(
            dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath
        )

        self.embed_dim = embed_dim
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.reg_factor = reg_factor
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
        self.module = BoxSquaredELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim=self.embed_dim,
            gamma=self.gamma,
            delta=self.delta,
            epsilon=self.epsilon,
            reg_factor=self.reg_factor,
            test_gci=self.test_gci,
        ).to(self.device)
        self.eval_method = self.module.eval_method

    def load_eval_data(self):
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
        self.init_model()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

    @property
    def new_test_set(self):
        self.load_eval_data()
        return self._new_test_set

    @property
    def dc_set(self):
        self.load_eval_data()
        return self._dc_set

    @property
    def training_set(self):
        self.load_eval_data()
        return self._training_set

    @property
    def testing_set(self):
        self.load_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        self.load_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        self.load_eval_data()
        return self._tail_entities

    def train(self):
        raise NotImplementedError


class BoxSquaredELModel(BoxSquaredEL):
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
                        l = th.mean(th.square(pos_loss))
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci0_neg, "gci0", neg=True)
                        )
                    cur_loss += l
                if len(gci1) > 0:
                    pos_loss = self.module(gci1, "gci1")
                    if "gci1" not in neg_types:
                        l = th.mean(th.square(pos_loss))
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci1_neg, "gci1", neg=True)
                        )
                    cur_loss += l
                if len(gci2) > 0:
                    pos_loss = self.module(gci2, "gci2")
                    if "gci2" not in neg_types:
                        l = th.mean(th.square(pos_loss))
                    else:
                        l = th.mean(th.square(pos_loss)) + th.mean(
                            th.square(self.module(gci2_neg, "gci2", neg=True))
                        )
                    cur_loss += l
                if len(gci3) > 0:
                    pos_loss = self.module(gci3, "gci3")
                    if "gci3" not in neg_types:
                        l = th.mean(th.square(pos_loss))
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci3_neg, "gci3", neg=True)
                        )
                    cur_loss += l
                if len(gci0_bot) > 0:
                    pos_loss = self.module(gci0_bot, "gci0_bot")
                    if "gci0_bot" not in neg_types:
                        l = th.mean(th.square(pos_loss))
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci0_bot_neg, "gci0_bot", neg=True)
                        )
                    cur_loss += l
                if len(gci1_bot) > 0:
                    pos_loss = self.module(gci1_bot, "gci1_bot")
                    if "gci1_bot" not in neg_types:
                        l = th.mean(th.square(pos_loss))
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci1_bot_neg, "gci1_bot", neg=True)
                        )
                    cur_loss += l
                if len(gci3_bot) > 0:
                    pos_loss = self.module(gci3_bot, "gci3_bot")
                    if "gci3_bot" not in neg_types:
                        l = th.mean(th.square(pos_loss))
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci3_bot_neg, "gci3_bot", neg=True)
                        )
                    cur_loss += l
                cur_loss += self.module.regularization_loss()
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
        return self.module.eval_method(data)
