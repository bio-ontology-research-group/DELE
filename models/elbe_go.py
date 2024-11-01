import mowl

mowl.init_jvm("8g", "1g", 8)

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELModule
from mowl.projection.factory import projector_factory
from losses.elbe_losses import *
from data_utils.dataloader_go import OntologyDataLoader
import torch as th
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import numpy as np
from mowl.datasets import PathDataset


class ELBEModule(ELModule):
    def __init__(
        self,
        nb_ont_classes,
        nb_rels,
        embed_dim=50,
        margin=0.1,
        epsilon=0.001,
    ):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.embed_dim = embed_dim

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(self.class_embed.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_embed.weight.data /= weight_data_normalized

        self.class_offset = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_offset.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(self.class_offset.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_offset.weight.data /= weight_data_normalized

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(
            self.rel_embed.weight.data, axis=1
        ).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data_normalized

        self.margin = margin
        self.epsilon = epsilon

    def gci0_loss(self, data, neg=False):
        return gci0_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            self.epsilon,
            neg=neg,
        )

    def gci0_bot_loss(self, data, neg=False):
        return gci0_bot_loss(
            data,
            self.class_offset,
            self.epsilon,
            neg=neg,
        )

    def gci1_loss(self, data, neg=False):
        return gci1_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            neg=neg,
        )

    def gci1_bot_loss(self, data, neg=False):
        return gci1_bot_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            self.epsilon,
            neg=neg,
        )

    def gci2_loss(self, data, neg=False, idxs_for_negs=None):
        return gci2_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
        )

    def gci3_loss(self, data, neg=False):
        return gci3_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
        )

    def gci3_bot_loss(self, data, neg=False):
        return gci3_bot_loss(
            data,
            self.class_offset,
            self.epsilon,
            neg=neg,
        )

    def eval_method(self, data):
        return gci2_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            neg=False,
        )


class ELBE(EmbeddingELModel):
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
        eval_property="http://interacts_with",
        path_to_dc=None,
        path_to_test=None,
    ):
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
        self.module = ELBEModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
            epsilon=self.epsilon,
        ).to(self.device)
        self.eval_method = self.module.eval_method

    def load_eval_data(self):
        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        eval_classes = self.dataset.evaluation_classes

        self._head_entities = eval_classes[0].as_str
        self._tail_entities = eval_classes[1].as_str

        eval_projector = projector_factory(
            "taxonomy_rels", taxonomy=False, relations=[eval_property]
        )
        
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


class ELBEModel(ELBE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def train(
        self,
        patience=10,
        epochs_no_improve=20,
        path_to_dc=None,
        prefix="4932.",
        neg_types=["gci2"],
        random_neg_fraction=1.0,
    ):
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        no_improve = 0
        best_loss = float("inf")

        go_classes = th.tensor(np.array([v for k, v in self.class_index_dict.items() if prefix not in k and "hing" not in k and "label" not in k])).to(self.device)
        protein_classes = th.tensor(np.array([v for k, v in self.class_index_dict.items() if prefix in k])).to(self.device)

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
                go_classes,
                protein_classes,
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
                go_classes,
                protein_classes,
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
                train_loss += cur_loss.detach().item()
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

            train_loss /= num_steps

            loss = 0
            with th.no_grad():
                self.module.eval()
                valid_loss = 0
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