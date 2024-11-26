import mowl

mowl.init_jvm("8g")

from mowl.datasets import PathDataset
from org.semanticweb.owlapi.model import OWLClass

"""
Helper functions from mOWL library
"""
class Entities:
    def __init__(self, collection):
        self._collection = self.check_owl_type(collection)
        self._collection = sorted(self._collection, key=lambda x: x.toStringID())
        self._name_owlobject = self.to_dict()
        self._index_dict = self.to_index_dict()

    def __getitem__(self, idx):
        return self._collection[idx]

    def __len__(self):
        return len(self._collection)

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind < len(self._collection):
            item = self._collection[self.ind]
            self.ind += 1
            return item
        raise StopIteration

    def check_owl_type(self, collection):
        """This method checks whether the elements in the provided collection
        are of the correct type.
        """
        raise NotImplementedError

    def to_str(self, owl_class):
        raise NotImplementedError

    def to_dict(self):
        """Generates a dictionaty indexed by OWL entities IRIs and the values
        are the corresponding OWL entities.
        """
        dict_ = {self.to_str(ent): ent for ent in self._collection}
        return dict_

    def to_index_dict(self):
        """Generates a dictionary indexed by OWL objects and the values
        are the corresponding indicies.
        """
        dict_ = {v: k for k, v in enumerate(self._collection)}
        return dict_

    @property
    def as_str(self):
        """Returns the list of entities as string names."""
        return list(self._name_owlobject.keys())

    @property
    def as_owl(self):
        """Returns the list of entities as OWL objects."""
        return list(self._name_owlobject.values())

    @property
    def as_dict(self):
        """Returns the dictionary of entities indexed by their names."""
        return self._name_owlobject

    @property
    def as_index_dict(self):
        """Returns the dictionary of entities indexed by their names."""
        return self._index_dict


class OWLClasses(Entities):
    """
    Iterable for :class:org.semanticweb.owlapi.model.OWLClass
    """

    def check_owl_type(self, collection):
        for item in collection:
            if not isinstance(item, OWLClass):
                raise TypeError("Type of elements in collection must be OWLClass.")
        return collection

    def to_str(self, owl_class):
        name = str(owl_class.toStringID())
        return name


class OWLTwoVecStarDataset(PathDataset):
    """
    OWL2Vec* dataset class
    """
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation"""

        if self._evaluation_classes is None:
            classes = set()
            for _, owl_cls in self.classes.as_dict.items():
                classes.add(owl_cls)
            self._evaluation_classes = OWLClasses(classes)

        return self._evaluation_classes

class PPIYeastDataset(PathDataset):
    """
    Yeast iw (interacts_with) dataset class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "4932." in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://interacts_with"


class AFPYeastDataset(PathDataset):
    """
    Yeast hf (has_function) dataset class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:
            proteins = set()
            gos = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "4932." in owl_name:
                    proteins.add(owl_cls)
                else:
                    if "label" not in owl_name and "hing" not in owl_name:
                        gos.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(gos)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://has_function"