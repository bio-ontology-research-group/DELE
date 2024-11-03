import mowl

mowl.init_jvm("150g")

from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from uk.ac.manchester.cs.owl.owlapi import OWLSubClassOfAxiomImpl
from java.util import HashSet
from mowl.ontology.normalize import ELNormalizer
from org.semanticweb.owlapi.model import IRI
import re
import numpy as np
import os
import regex
from mowl.owlapi.defaults import TOP, BOT


def precompute_gci0_dc(data_path, ontology_file):
    """
    Compute GCI0 (`C \sqsubseteq D`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, ontology_file))

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager

    reasoner_factory = ElkReasonerFactory()
    reasoner = reasoner_factory.createReasoner(dataset.ontology)
    new_gci0_axioms = HashSet()

    gci0_dict = {k: [k] for k in list(dataset.ontology.getClassesInSignature())}
    for cl in list(dataset.ontology.getClassesInSignature()):
        superclasses = list(reasoner.getSuperClasses(cl, False).getFlattened())
        subclasses = list(reasoner.getSubClasses(cl, False).getFlattened())
        for elem in superclasses:
            if elem not in gci0_dict[cl]:
                gci0_dict[cl].append(elem)
        for elem in subclasses:
            try:
                if cl not in gci0_dict[elem]:
                    gci0_dict[elem].append(cl)
            except:
                if elem not in gci0_dict.keys():
                    gci0_dict[elem] = [cl]
                else:
                    gci0_dict[elem].append(cl)

    for cl in gci0_dict.keys():
        for super_cl in gci0_dict[cl]:
            if 'Nothing' in str(cl):
                cl = adapter.create_class(BOT)
            if 'Thing' in str(cl):
                cl = adapter.create_class(TOP)
            if 'Nothing' in str(super_cl):
                super_cl = adapter.create_class(BOT)
            if 'Thing' in str(super_cl):
                super_cl = adapter.create_class(TOP)
            new_gci0_axioms.add(OWLSubClassOfAxiomImpl(cl, super_cl, []))

    for ax in list(train_norm["gci0"]):
        new_gci0_axioms.add(ax.owl_axiom)

    new_gci0_train = adapter.create_ontology("http://gci0_ontology")
    manager.addAxioms(new_gci0_train, new_gci0_axioms)
    adapter.owl_manager.saveOntology(
        new_gci0_train,
        IRI.create("file://" + os.path.join(data_path, "gci0_ontology.owl")),
    )
    return


def get_gci0_dict(data_path):
    """
    Create a dictionary `class: its superclasses`

    :param data_path: absolute filepath to the folder containing ontology with GCI0 deductive closure
    :type data_path: str
    :return gci0_dict: a dictionary `class: its superclasses`
    :type gci0_dict: dict(org.semanticweb.owlapi.model.OWLClass, list(org.semanticweb.owlapi.model.OWLClass))
    """

    gci0_dataset = PathDataset(os.path.join(data_path, "gci0_ontology.owl"))
    gci0_dict = {}
    for ax in list(gci0_dataset.ontology.getAxioms()):
        if "SubClassOf" not in str(ax):
            continue
        else:
            classes = list(ax.getClassesInSignature())
            str_ax = re.split("SubClassOf| ", str(ax))[1:]
            if str(classes[0]) == str_ax[0][1:]:
                if classes[0] not in gci0_dict.keys():
                    try:
                        gci0_dict[classes[0]] = [classes[1]]
                    except:
                        gci0_dict[classes[0]] = [classes[0]]
                else:
                    try:
                        gci0_dict[classes[0]].append(classes[1])
                    except:
                        gci0_dict[classes[0]].append(classes[0])
            elif str(classes[1]) == str_ax[0][1:]:
                if classes[1] not in gci0_dict.keys():
                    gci0_dict[classes[1]] = [classes[0]]
                else:
                    gci0_dict[classes[1]].append(classes[0])
    return gci0_dict


def get_inv_gci0_dict(gci0_dict):
    """
    Create a dictionary `class: its subclasses`

    :param gci0_dict: dictionary `class: its superclasses`
    :type gci0_dict: dict(org.semanticweb.owlapi.model.OWLClass, list(org.semanticweb.owlapi.model.OWLClass))
    :return inv_gci0_dict: a dictionary `class: its subclasses`
    :type inv_gci0_dict: dict(org.semanticweb.owlapi.model.OWLClass, list(org.semanticweb.owlapi.model.OWLClass))
    """

    inv_gci0_dict = {}
    for k in gci0_dict.keys():
        for v in gci0_dict[k]:
            if v not in inv_gci0_dict.keys():
                inv_gci0_dict[v] = [k]
            else:
                inv_gci0_dict[v].append(k)
    return inv_gci0_dict


def precompute_gci1_dc_1(data_path, ontology_file):
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, ontology_file))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci1_axioms = HashSet()
    gci1_extracted = [elem.owl_axiom for elem in train_norm["gci1"]]
    gci1_bot_extracted = [elem.owl_axiom for elem in train_norm["gci1_bot"]]
    all_classes = list(dataset.ontology.getClassesInSignature())

    for ax in gci1_extracted:

        str_ax = re.split("SubClassOf\((.*)\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ''][0]
        matches = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax)]
        c_d, E = matches[0], adapter.create_class(matches[1][1:-1])
        str_ax_2 = re.split("ObjectIntersectionOf\((.*)\)", c_d)
        str_ax_2 = [elem for elem in str_ax_2 if elem != ''][0]
        matches_2 = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax_2)]
        C, D = adapter.create_class(matches_2[0][1:-1]), adapter.create_class(matches_2[1][1:-1])
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
        else:
            C_subclasses = [C]
        if D in inv_gci0_dict.keys():
            D_subclasses = inv_gci0_dict[D]
        else:
            D_subclasses = [D]
        if E in gci0_dict.keys():
            E_superclasses = gci0_dict[E]
        else:
            E_superclasses = [E]
        for c_subclass in C_subclasses:
            for d_subclass in D_subclasses:
                for e_superclass in E_superclasses:
                        new_gci1_axioms.add(
                            adapter.create_subclass_of(
                                adapter.create_object_intersection_of(c_subclass, d_subclass), e_superclass
                            )
                        )
        
    for ax in gci1_bot_extracted:

        str_ax = re.split("SubClassOf\((.*)\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ''][0]
        matches = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax)]
        c_d = matches[0]
        str_ax_2 = re.split("ObjectIntersectionOf\((.*)\)", c_d)
        str_ax_2 = [elem for elem in str_ax_2 if elem != ''][0]
        matches_2 = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax_2)]
        C, D = adapter.create_class(matches_2[0][1:-1]), adapter.create_class(matches_2[1][1:-1])
        for cl in all_classes:
            if 'Nothing' in str(cl):
                cl = adapter.create_class(BOT)
            if 'Thing' in str(cl):
                cl = adapter.create_class(TOP)
            new_gci1_axioms.add(
                adapter.create_subclass_of(
                    adapter.create_object_intersection_of(C, D), cl
                )
            )

    new_gci1_train = adapter.create_ontology("http://gci1_ontology")
    manager.addAxioms(new_gci1_train, new_gci1_axioms)
    adapter.owl_manager.saveOntology(
        new_gci1_train,
        IRI.create("file://" + os.path.join(data_path, "gci1_ontology_1.owl")),
    )
    return


def precompute_gci1_dc_2(data_path, ontology_file):
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, ontology_file))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci1_axioms = HashSet()
    gci1_extracted = [elem.owl_axiom for elem in train_norm["gci1"]]
    gci1_bot_extracted = [elem.owl_axiom for elem in train_norm["gci1_bot"]]
    all_classes = list(dataset.ontology.getClassesInSignature())
        
    for ax in gci1_bot_extracted:

        str_ax = re.split("SubClassOf\((.*)\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ''][0]
        matches = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax)]
        c_d = matches[0]
        str_ax_2 = re.split("ObjectIntersectionOf\((.*)\)", c_d)
        str_ax_2 = [elem for elem in str_ax_2 if elem != ''][0]
        matches_2 = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax_2)]
        C, D = adapter.create_class(matches_2[0][1:-1]), adapter.create_class(matches_2[1][1:-1])
        for cl in all_classes:
            if 'Nothing' in str(cl):
                cl = adapter.create_class(BOT)
            if 'Thing' in str(cl):
                cl = adapter.create_class(TOP)
            new_gci1_axioms.add(
                adapter.create_subclass_of(
                    adapter.create_object_intersection_of(C, D), cl
                )
            )

    for cl1 in all_classes:
        if 'Nothing' in str(cl1):
            cl1 = adapter.create_class(BOT)
        if 'Thing' in str(cl1):
            cl1 = adapter.create_class(TOP)
        for cl2 in all_classes:
            if 'Nothing' in str(cl2):
                cl2 = adapter.create_class(BOT)
            if 'Thing' in str(cl2):
                cl2 = adapter.create_class(TOP)
            new_gci1_axioms.add(
                adapter.create_subclass_of(
                    adapter.create_object_intersection_of(cl1, adapter.create_class(BOT)), cl2
                )
            )

    if adapter.create_class(BOT) in list(inv_gci0_dict.keys()):
        for cl in inv_gci0_dict[adapter.create_class(BOT)]:
            if 'Nothing' in str(cl):
                cl = adapter.create_class(BOT)
            if 'Thing' in str(cl):
                cl = adapter.create_class(TOP)
            for cl1 in all_classes:
                if 'Nothing' in str(cl1):
                    cl1 = adapter.create_class(BOT)
                if 'Thing' in str(cl1):
                    cl1 = adapter.create_class(TOP)
                for cl2 in all_classes:
                    if 'Nothing' in str(cl2):
                        cl2 = adapter.create_class(BOT)
                    if 'Thing' in str(cl2):
                        cl2 = adapter.create_class(TOP)
                    new_gci1_axioms.add(
                        adapter.create_subclass_of(
                            adapter.create_object_intersection_of(cl1, cl), cl2
                        )
                    )

    for cl1 in all_classes:
        if 'Nothing' in str(cl1):
            cl1 = adapter.create_class(BOT)
        if 'Thing' in str(cl1):
            cl1 = adapter.create_class(TOP)
        for cl2 in all_classes:
            if 'Nothing' in str(cl2):
                cl2 = adapter.create_class(BOT)
            if 'Thing' in str(cl2):
                cl2 = adapter.create_class(TOP)
            if cl2 in list(gci0_dict.keys()):
                superclasses_cl2 = gci0_dict[cl2]
            else:
                superclasses_cl2 = [cl2]
            for superclass_cl2 in superclasses_cl2:
                new_gci1_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_intersection_of(cl1, cl2), superclass_cl2
                    )
                )

    for cl in list(inv_gci0_dict.keys()):
        if cl in gci0_dict.keys():
            cl_superclasses = gci0_dict[cl]
        else:
            cl_superclasses = [cl]
        cl_subclasses = inv_gci0_dict[cl]
        for cl1 in cl_subclasses:
            if cl1 in inv_gci0_dict.keys():
                cl1_subclasses = inv_gci0_dict[cl1]
            else:
                cl1_subclasses = [cl1]
            for cl2 in cl_subclasses:
                if cl2 in inv_gci0_dict.keys():
                    cl2_subclasses = inv_gci0_dict[cl2]
                else:
                    cl2_subclasses = [cl2]
                for cl1_subclass in cl1_subclasses:
                    for cl2_subclass in cl2_subclasses:
                        for cl_superclass in cl_superclasses:
                            new_gci1_axioms.add(
                                adapter.create_subclass_of(
                                    adapter.create_object_intersection_of(cl1_subclass, cl2_subclass), cl_superclass
                                )
                            )

    for cl in list(gci0_dict.keys()):
        cl_superclasses = gci0_dict[cl]
        for cl_superclass in cl_superclasses:
            new_gci1_axioms.add(
                adapter.create_subclass_of(
                    adapter.create_object_intersection_of(cl, adapter.create_class(TOP)), cl_superclass
                )
            )

    new_gci1_train = adapter.create_ontology("http://gci1_ontology")
    manager.addAxioms(new_gci1_train, new_gci1_axioms)
    adapter.owl_manager.saveOntology(
        new_gci1_train,
        IRI.create("file://" + os.path.join(data_path, "gci1_ontology_2.owl")),
    )
    return


def precompute_gci2_dc(data_path, ontology_file):
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, ontology_file))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci2_axioms = HashSet()
    gci2_extracted = [elem.owl_axiom for elem in train_norm["gci2"]]
    all_rels = list(dataset.ontology.getObjectPropertiesInSignature())
    all_classes = list(dataset.ontology.getClassesInSignature())

    for ax in gci2_extracted:

        str_ax = re.split("SubClassOf\((.*)\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ''][0]
        matches = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax)]
        C, r_d = adapter.create_class(matches[0][1:-1]), matches[1]
        str_ax_2 = re.split("ObjectSomeValuesFrom\((.*)\)", r_d)
        str_ax_2 = [elem for elem in str_ax_2 if elem != ''][0]
        matches_2 = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax_2)]
        R, D = adapter.create_object_property(matches_2[0][1:-1]), adapter.create_class(matches_2[1][1:-1])
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
        else:
            C_subclasses = [C]
        if D in gci0_dict.keys():
            D_superclasses = gci0_dict[D]
        else:
            D_superclasses = [D]
        for c_subclass in C_subclasses:
            for d_superclass in D_superclasses:
                new_gci2_axioms.add(
                    adapter.create_subclass_of(
                        c_subclass, adapter.create_object_some_values_from(R, d_superclass)
                    )
                )

    for cl in all_classes:
        if 'Nothing' in str(cl):
            continue
        if 'Thing' in str(cl):
            cl = adapter.create_class(TOP)
        for r in all_rels:
            new_gci2_axioms.add(
                adapter.create_subclass_of(
                    adapter.create_class(BOT), adapter.create_object_some_values_from(r, cl)
                )
            )

    if adapter.create_class(BOT) in list(inv_gci0_dict.keys()):
        for cl in inv_gci0_dict[adapter.create_class(BOT)]:
            if 'Nothing' in str(cl):
                cl = adapter.create_class(BOT)
            if 'Thing' in str(cl):
                cl = adapter.create_class(TOP)
            for cl2 in all_classes:
                if 'Nothing' in str(cl2):
                    continue
                if 'Thing' in str(cl2):
                    cl2 = adapter.create_class(TOP)
                for r in all_rels:
                    new_gci2_axioms.add(
                        adapter.create_subclass_of(
                            cl, adapter.create_object_some_values_from(r, cl2)
                        )
                    )

    new_gci2_train = adapter.create_ontology("http://gci2_ontology")
    manager.addAxioms(new_gci2_train, new_gci2_axioms)
    adapter.owl_manager.saveOntology(
        new_gci2_train,
        IRI.create("file://" + os.path.join(data_path, "gci2_ontology.owl")),
    )
    return


def precompute_gci3_dc(data_path, ontology_file):
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, ontology_file))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci3_axioms = HashSet()
    gci3_extracted = [elem.owl_axiom for elem in train_norm["gci3"]]

    for ax in gci3_extracted:

        str_ax = re.split("SubClassOf\((.*)\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ''][0]
        matches = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax)]
        r_c, D = matches[0], adapter.create_class(matches[1][1:-1])
        str_ax_2 = re.split("ObjectSomeValuesFrom\((.*)\)", r_c)
        str_ax_2 = [elem for elem in str_ax_2 if elem != ''][0]
        matches_2 = [match.group() for match in regex.finditer(r"(?:(\((?>[^()]+|(?1))*\))|\S)+", str_ax_2)]
        R, C = adapter.create_object_property(matches_2[0][1:-1]), adapter.create_class(matches_2[1][1:-1])
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
        else:
            C_subclasses = [C]
        if D in gci0_dict.keys():
            D_superclasses = gci0_dict[D]
        else:
            D_superclasses = [D]
        for c_subclass in C_subclasses:
            for d_superclass in D_superclasses:
                new_gci3_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_some_values_from(R, c_subclass), d_superclass
                    )
                )

    for cl in all_classes:
        if 'Nothing' in str(cl):
            continue
        if 'Thing' in str(cl):
            cl = adapter.create_class(TOP)
        for r in all_rels:
            new_gci3_axioms.add(
                adapter.create_subclass_of(
                    adapter.create_object_some_values_from(r, cl), adapter.create_class(TOP)
                )
            )

    new_gci3_train = adapter.create_ontology("http://gci3_ontology")
    manager.addAxioms(new_gci3_train, new_gci3_axioms)
    adapter.owl_manager.saveOntology(
        new_gci3_train,
        IRI.create("file://" + os.path.join(data_path, "gci3_ontology.owl")),
    )
    return


def precompute_gci1_bot_dc(data_path, ontology_file):
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, ontology_file))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci1_bot_axioms = HashSet()
    gci1_bot_extracted = [elem.owl_axiom for elem in train_norm["gci1_bot"]]

    for ax in gci1_bot_extracted:

        classes = np.array(list(ax.getClassesInSignature()))
        str_ax = re.split("SubClassOf|ObjectIntersectionOf|\(| |\)|\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ""]
        ixs = []
        for cl in classes:
            ixs.append(str_ax.index(str(cl)))
        classes = classes[ixs]
        C = classes[0]
        D = classes[1]
        if 'Nothing' in str(C):
            C = adapter.create_class(BOT)
        if 'Thing' in str(C):
            C = adapter.create_class(TOP)
        if 'Nothing' in str(D):
            D = adapter.create_class(BOT)
        if 'Thing' in str(D):
            D = adapter.create_class(TOP)
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
        else:
            C_subclasses = [C]
        if D in inv_gci0_dict.keys():
            D_subclasses = inv_gci0_dict[D]
        else:
            D_subclasses = [D]
        for c_subclass in C_subclasses:
            for d_subclass in D_subclasses:
                new_gci1_bot_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_intersection_of(c_subclass, D_subclass), E
                    )
                )

    new_gci1_bot_train = adapter.create_ontology("http://gci1_bot_ontology")
    manager.addAxioms(new_gci1_bot_train, new_gci1_bot_axioms)
    adapter.owl_manager.saveOntology(
        new_gci1_bot_train,
        IRI.create("file://" + os.path.join(data_path, "gci1_bot_ontology.owl")),
    )
    return