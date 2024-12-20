import mowl

mowl.init_jvm("8g")

from mowl.evaluation.base import Evaluator, compute_rank_roc

import torch as th
import logging
from tqdm import tqdm
from scipy.stats import rankdata
import numpy as np

th.manual_seed(0)


class RankBasedEvaluator(Evaluator):

    """
    This class corresponds to evaluation based on ranking. That is, for each testing triple \
    :math:`(h,r,t)`, scores are computed for triples :math:`(h,r,t')` for all possible \
    :math:`t'`. After that, the ranking of the testing triple :math:`(h,r,t)` score is obtained.

    :param class_index_emb: dictionary of classes and their embeddings
    :type class_index_emb: dict(str, np.array)
    :param relation_index_emb: dictionary of relations and their embeddings
    :type relation_index_emb: dict(str, np.array)
    :param testing_set: Set of triples that are true positives.
    :type testing_set: list(mowl.projection.edge.Edge)
    :param eval_method: evaluation method score the triples
    :type eval_method: function
    :param training_set: Set of triples that are true positives but exist in the training set. \
This is used to compute filtered metrics.
    :type training_set: list(mowl.projection.edge.Edge)
    :param head_entities: List of entities that are used as head entities in the testing set.
    :type head_entities: list(str)
    :param tail_entities: List of entities that are used as tail entities in the testing set.
    :type tail_entities: list(str)
    :param device: Use `cpu` or `cuda`
    :type device: str
    """

    def __init__(
        self,
        class_index_emb,
        relation_index_emb,
        testing_set,
        eval_method,
        training_set,
        head_entities,
        tail_entities,
        device,
        auc_mode="macro",
        test_gci="gci2",
    ):
        super().__init__(device)

        self.eval_method = eval_method
        self.compute_filtered_metrics = True
        if training_set is None:
            self.compute_filtered_metrics = False
            logging.info(
                "Training set was not input. Filtered metrics will not be available."
            )

        self.device = device
        self.auc_mode = auc_mode
        self.test_gci = test_gci
        self._data_loaded: bool

        self.relation_index_emb = relation_index_emb

        self.head_entities = head_entities
        self.head_name_indexemb: dict
        self.head_indexemb_indexsc: dict

        self.tail_entities = tail_entities
        self.tail_name_indexemb: dict
        self.tail_indexemb_indexsc: dict

        self.class_index_emb = class_index_emb
        self.training_set = [x.astuple() for x in training_set]
        self.testing_set = [x.astuple() for x in testing_set]

        self._loaded_ht_data = False
        self._loaded_tr_scores = False

        self.filter_head_tail_data()

        self.training_scores = np.ones(
            (len(self.head_entities), len(self.tail_entities)), dtype=np.int32
        )
        self.testing_scores = np.ones(
            (len(self.head_entities), len(self.tail_entities)), dtype=np.int32
        )
        self.testing_predictions = np.zeros(
            (len(self.head_entities), len(self.tail_entities)), dtype=np.int32
        )

        self.load_training_scores()

    def filter_head_tail_data(self):
        if self._loaded_ht_data:
            return

        new_head_entities = set()
        new_tail_entities = set()

        for e in self.head_entities:
            if e in self.class_index_emb:
                new_head_entities.add(e)
            else:
                logging.info(
                    "Entity %s not present in the embeddings dictionary. Ignoring it.",
                    e,
                )

        for e in self.tail_entities:
            if e in self.class_index_emb:
                new_tail_entities.add(e)
            else:
                logging.info(
                    "Entity %s not present in the embeddings dictionary. Ignoring it.",
                    e,
                )

        self.head_entities = new_head_entities
        self.tail_entities = new_tail_entities

        self.head_name_indexemb = {
            k: self.class_index_emb[k] for k in self.head_entities
        }
        self.tail_name_indexemb = {
            k: self.class_index_emb[k] for k in self.tail_entities
        }

        self.head_indexemb_indexsc = {
            v: k for k, v in enumerate(self.head_name_indexemb.values())
        }
        self.tail_indexemb_indexsc = {
            v: k for k, v in enumerate(self.tail_name_indexemb.values())
        }

        self._loaded_ht_data = True

    def load_training_scores(self):
        if self._loaded_tr_scores or not self.compute_filtered_metrics:
            return
        
        for c, _, d in self.training_set:
            if (c not in self.head_entities) or not (d in self.tail_entities):
                continue

            c, d = self.head_name_indexemb[c], self.tail_name_indexemb[d]
            c, d = self.head_indexemb_indexsc[c], self.tail_indexemb_indexsc[d]

            self.training_scores[c, d] = 1000000

        logging.info("Training scores created")
        self._loaded_tr_scores = True

    def evaluate(self, activation=None, show=False):
        if activation is None:

            def activation(x):
                return x

        top1 = 0
        top10 = 0
        top100 = 0
        mean_rank = 0
        ftop1 = 0
        ftop10 = 0
        ftop100 = 0
        fmean_rank = 0
        ranks = {} 
        franks = {}
        if self.auc_mode == "micro":
            micro_ranks = {}
            micro_franks = {}
            #
            micro_mean_ranks = {}
            micro_mean_franks = {}

        num_tail_entities = len(self.tail_entities)

        worst_rank = num_tail_entities

        n = len(self.testing_set)

        if self.test_gci == "gci0":
            for c, _, d in tqdm(self.testing_set):
                if not (c in self.head_entities) or not (d in self.tail_entities):
                    n -= 1
                    if d not in self.tail_entities:
                        worst_rank -= 1
                    continue

                # Embedding indices
                c_emb_idx, d_emb_idx = (
                    self.head_name_indexemb[c],
                    self.tail_name_indexemb[d],
                )

                # Scores matrix labels
                c_sc_idx = self.head_indexemb_indexsc[c_emb_idx]
                d_sc_idx = self.tail_indexemb_indexsc[d_emb_idx]

                data = [
                    [c_emb_idx, self.tail_name_indexemb[x]] for x in self.tail_entities
                ]
                data = np.array(data)
                data = th.tensor(data, requires_grad=False).to(self.device)
                with th.no_grad():
                    res = self.eval_method(data)
                    res = activation(res)
                    res = res.squeeze().cpu().detach().numpy()

                self.testing_predictions[c_sc_idx, :] = res
                index = rankdata(res, method="average")
                rank = index[d_sc_idx]

                if rank == 1:
                    top1 += 1
                if rank <= 10:
                    top10 += 1
                if rank <= 100:
                    top100 += 1

                mean_rank += rank
                if rank not in ranks:
                    ranks[rank] = 0
                ranks[rank] += 1

                if self.auc_mode == "micro":
                    if c_sc_idx not in micro_ranks:
                        micro_ranks[c_sc_idx] = {rank: 1}
                    else:
                        if rank not in micro_ranks[c_sc_idx]:
                            micro_ranks[c_sc_idx][rank] = 0
                        micro_ranks[c_sc_idx][rank] += 1

                    if c_sc_idx not in micro_mean_ranks:
                        micro_mean_ranks[c_sc_idx] = 0
                    micro_mean_ranks[c_sc_idx] += rank

                # Filtered rank

                if self.compute_filtered_metrics:
                    fres = np.zeros_like(res)
                    count = 0
                    for ix in range(len(res)):
                        if res[ix] < 0 and self.training_scores[c_sc_idx, ix] == 1000000:
                            fres[ix] = -1000000 * res[ix]
                        elif res[ix] >= 0 and self.training_scores[c_sc_idx, ix] == 1000000:
                            fres[ix] = 1000000 * res[ix]
                            count += 1
                        else:
                            fres[ix] = res[ix]
                    index = rankdata(fres, method="average")
                    frank = index[d_sc_idx]

                    if frank == 1:
                        ftop1 += 1
                    if frank <= 10:
                        ftop10 += 1
                    if frank <= 100:
                        ftop100 += 1
                    fmean_rank += frank

                    if frank not in franks:
                        franks[frank] = 0
                    franks[frank] += 1

                    if self.auc_mode == "micro":
                        if c_sc_idx not in micro_franks:
                            micro_franks[c_sc_idx] = {frank: 1}
                        else:
                            if frank not in micro_franks[c_sc_idx]:
                                micro_franks[c_sc_idx][frank] = 0
                            micro_franks[c_sc_idx][frank] += 1

                        if c_sc_idx not in micro_mean_franks:
                            micro_mean_franks[c_sc_idx] = 0
                        micro_mean_franks[c_sc_idx] += frank

        elif self.test_gci == "gci2":
            for c, r, d in tqdm(self.testing_set):
                if not (c in self.head_entities) or not (d in self.tail_entities):
                    n -= 1
                    if d not in self.tail_entities:
                        worst_rank -= 1
                    continue

                # Embedding indices
                c_emb_idx, d_emb_idx = (
                    self.head_name_indexemb[c],
                    self.tail_name_indexemb[d],
                )

                # Scores matrix labels
                c_sc_idx = self.head_indexemb_indexsc[c_emb_idx]
                d_sc_idx = self.tail_indexemb_indexsc[d_emb_idx]

                r = self.relation_index_emb[r]

                data = [
                    [c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities
                ]
                data = np.array(data)
                data = th.tensor(data, requires_grad=False).to(self.device)
                with th.no_grad():
                    res = self.eval_method(data)
                    res = activation(res)
                    res = res.squeeze().cpu().detach().numpy()

                self.testing_predictions[c_sc_idx, :] = res
                index = rankdata(res, method="average")
                rank = index[d_sc_idx]

                if rank == 1:
                    top1 += 1
                if rank <= 10:
                    top10 += 1
                if rank <= 100:
                    top100 += 1

                mean_rank += rank
                if rank not in ranks:
                    ranks[rank] = 0
                ranks[rank] += 1

                if self.auc_mode == "micro":
                    if c_sc_idx not in micro_ranks:
                        micro_ranks[c_sc_idx] = {rank: 1}
                    else:
                        if rank not in micro_ranks[c_sc_idx]:
                            micro_ranks[c_sc_idx][rank] = 0
                        micro_ranks[c_sc_idx][rank] += 1

                    if c_sc_idx not in micro_mean_ranks:
                        micro_mean_ranks[c_sc_idx] = 0
                    micro_mean_ranks[c_sc_idx] += rank

                # Filtered rank

                if self.compute_filtered_metrics:
                    fres = np.zeros_like(res)
                    count = 0
                    for ix in range(len(res)):
                        if res[ix] < 0 and self.training_scores[c_sc_idx, ix] == 1000000:
                            fres[ix] = -1000000 * res[ix]
                        elif res[ix] >= 0 and self.training_scores[c_sc_idx, ix] == 1000000:
                            fres[ix] = 1000000 * res[ix]
                            count += 1
                        else:
                            fres[ix] = res[ix]
                    index = rankdata(fres, method="average")
                    frank = index[d_sc_idx]

                    if frank == 1:
                        ftop1 += 1
                    if frank <= 10:
                        ftop10 += 1
                    if frank <= 100:
                        ftop100 += 1
                    fmean_rank += frank

                    if frank not in franks:
                        franks[frank] = 0
                    franks[frank] += 1

                    if self.auc_mode == "micro":
                        if c_sc_idx not in micro_franks:
                            micro_franks[c_sc_idx] = {frank: 1}
                        else:
                            if frank not in micro_franks[c_sc_idx]:
                                micro_franks[c_sc_idx][frank] = 0
                            micro_franks[c_sc_idx][frank] += 1

                        if c_sc_idx not in micro_mean_franks:
                            micro_mean_franks[c_sc_idx] = 0
                        micro_mean_franks[c_sc_idx] += frank

        else:
            raise NotImplementedError

        top1 /= n
        top10 /= n
        top100 /= n
        mean_rank /= n

        ftop1 /= n
        ftop10 /= n
        ftop100 /= n
        fmean_rank /= n

        auc_x = list(ranks.keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        sum_rank = sum(ranks.values())  # number of evaluation points

        for x in auc_x:
            tpr += ranks[x]
            auc_y.append(tpr / sum_rank)

        auc_x.append(worst_rank)
        auc_y.append(1)

        rank_auc = compute_rank_roc(ranks, worst_rank)
        frank_auc = compute_rank_roc(franks, worst_rank)

        if self.auc_mode == "micro":
            micro_aucs = []
            for k in micro_ranks:
                micro_aucs.append(compute_rank_roc(micro_ranks[k], worst_rank))
            micro_auc = sum(micro_aucs) / len(micro_aucs)

            micro_faucs = []
            for k in micro_franks:
                micro_faucs.append(compute_rank_roc(micro_franks[k], worst_rank))
            micro_fauc = sum(micro_faucs) / len(micro_faucs)

            micro_mean_rank_arr = [micro_mean_ranks[k] / sum(micro_ranks[k].values()) for k in micro_mean_ranks.keys()]
            micro_mean_rank = sum(micro_mean_rank_arr) / len(micro_mean_rank_arr)
            micro_fmean_rank_arr = [micro_mean_franks[k] / sum(micro_franks[k].values()) for k in micro_mean_franks.keys()]
            micro_fmean_rank = sum(micro_fmean_rank_arr) / len(micro_fmean_rank_arr)

        if show and self.auc_mode == "macro":
            print(f"Hits@1:    {top1:.2f} Filtered:   {ftop1:.2f}")
            print(f"Hits@10:   {top10:.2f} Filtered:   {ftop10:.2f}")
            print(f"Hits@100:  {top100:.2f} Filtered:   {ftop100:.2f}")
            print(f"MR:        {mean_rank:.2f} Filtered: {fmean_rank:.2f}")
            print(f"Macro AUC: {rank_auc:.2f} Filtered:   {frank_auc:.2f}")

        elif show and self.auc_mode == "micro":
            print(f"Hits@1:    {top1:.2f} Filtered:   {ftop1:.2f}")
            print(f"Hits@10:   {top10:.2f} Filtered:   {ftop10:.2f}")
            print(f"Hits@100:  {top100:.2f} Filtered:   {ftop100:.2f}")
            print(f"Macro MR:  {mean_rank:.2f} Filtered: {fmean_rank:.2f}")
            print(f"Micro MR:  {micro_mean_rank:.2f} Filtered: {micro_fmean_rank:.2f}")
            print(f"Macro AUC: {rank_auc:.2f} Filtered:   {frank_auc:.2f}")
            print(f"Micro AUC: {micro_auc:.2f} Filtered:   {micro_fauc:.2f}")

        if self.auc_mode == "macro":
            self.metrics = {
                "hits@1": top1,
                "fhits@1": ftop1,
                "hits@10": top10,
                "fhits@10": ftop10,
                "hits@100": top100,
                "fhits@100": ftop100,
                "mean_rank": mean_rank,
                "fmean_rank": fmean_rank,
                "rank_auc": rank_auc,
                "frank_auc": frank_auc,
            }

        elif self.auc_mode == "micro":
            self.metrics = {
                "hits@1": top1,
                "fhits@1": ftop1,
                "hits@10": top10,
                "fhits@10": ftop10,
                "hits@100": top100,
                "fhits@100": ftop100,
                "mean_rank": mean_rank,
                "micro_mean_rank": micro_mean_rank,
                "fmean_rank": fmean_rank,
                "micro_fmean_rank": micro_fmean_rank,
                "macro_rank_auc": rank_auc,
                "macro_frank_auc": frank_auc,
                "micro_rank_auc": micro_auc,
                "micro_frank_auc": micro_fauc,
            }

        print('Evaluation finished. Access the results using the "metrics" attribute.')


class ModelRankBasedEvaluator(RankBasedEvaluator):
    """This class corresponds to evaluation based on ranking, where the embedding information of \
an entity is enclosed in some model.

    :param model: The model to be evaluated.
    :type model: mowl.base_models.EmbeddingModel
    :param device: The device to be used for evaluation. Defaults to 'cpu'.
    :type device: str, optional
    :param eval_method: The method used for the evaluation. If None, the method will be set \
to ``self.eval_method``. Defaults to None.
    :type eval_method: callable, optional
"""

    def __init__(self, model, device="cpu", eval_method=None, auc_mode="macro", test_gci="gci2", filter_by="train", test="test"):
        self.model = model
        self.model.load_best_model()

        class_index_emb = self.model.class_index_dict
        
        if test == "test":
            testing_set = self.model.testing_set
        else:
            testing_set = self.model.new_test_set
        if filter_by == "train":
            training_set = self.model.training_set
        else:
            training_set = self.model.dc_set
        head_entities = self.model.head_entities
        tail_entities = self.model.tail_entities
        eval_method = self.model.eval_method if eval_method is None else eval_method

        relation_index_emb = self.model.object_property_index_dict

        super().__init__(
            class_index_emb,
            relation_index_emb,
            testing_set,
            eval_method,
            training_set,
            head_entities,
            tail_entities,
            device,
            auc_mode,
            test_gci,
        )