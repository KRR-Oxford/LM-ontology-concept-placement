# reconstruct owl by changing equiv axioms to subclasses
# using BLINKout+/ontologies/SNOMEDCT-US-20140901.owl
# using deepOnto (python 3.8)
# install to latest commit
#pip install git+https://github.com/KRR-Oxford/DeepOnto.git@4b750938ca01d65d35c25a988a443c95ef9ff5b2

from tqdm import tqdm
from deeponto import init_jvm
init_jvm("32g")
from deeponto.onto import Ontology
#from deeponto.onto.ontology import get_equivalence_axioms,get_subsumption_axioms

#load ontology
ontology_name = 'SNOMEDCT-US-20140901.owl'
#ontology_name = 'SNOMEDCT-US-20170301.owl'
#ontology_name = 'SNOMEDCT-US-201907.owl'
fn_ontology = "../ontologies/%s" % ontology_name 
onto_sno = Ontology(fn_ontology)

list_equiv_axioms = onto_sno.get_equivalence_axioms(entity_type="Classes")
print('list_equiv_axioms:',len(list_equiv_axioms))

for eq in tqdm(list_equiv_axioms):
    #print(eq)
    gcis = list(eq.asOWLSubClassOfAxioms())
    for gci in gcis: #TODO: only add when sub_class is an atomic class, i.e. super_class a complex class?
        super_class = gci.getSuperClass()
        sub_class = gci.getSubClass()
        super_class_conjunct_set = list(super_class.asConjunctSet())
        #print('super_class_conjunct_set:',super_class_conjunct_set)
        #print('super_class_first:',super_class_conjunct_set[0])
        if len(super_class_conjunct_set) == 1:
            # if super_class only has one conjunct part (i.e. atomic)
            continue
        if 'EquivalentClasses(<http://snomed.info/id/239539008>' in str(eq):
            print('gci:',gci)
            print('super_class:',super_class,'\nsub_class:',sub_class)
            print('super_class_conjunct_set:',super_class_conjunct_set)
    
        for super_class_conjunct_part in super_class_conjunct_set:
            sub_axiom = onto_sno.owl_data_factory.getOWLSubClassOfAxiom(sub_class, super_class_conjunct_part)
            onto_sno.add_axiom(sub_axiom)
            if 'EquivalentClasses(<http://snomed.info/id/239539008>' in str(eq):
                print('super_class_conjunct_part:',super_class_conjunct_part)
        
list_sub_axioms = onto_sno.get_subsumption_axioms(entity_type="Classes")
print('list_sub_axioms:',len(list_sub_axioms))

for sub in tqdm(list_sub_axioms):
    #     print(sub)
    super_class = sub.getSuperClass()
    sub_class = sub.getSubClass()
    super_class_conjunct_set = list(super_class.asConjunctSet())
    for super_class_conjunct_part in super_class_conjunct_set:
        sub_axiom = onto_sno.owl_data_factory.getOWLSubClassOfAxiom(sub_class, super_class_conjunct_part)
        onto_sno.add_axiom(sub_axiom)

ontology_name_prefix = ontology_name.split('.owl')[0]
onto_sno.save_onto("../ontologies/%s-rev.owl" % ontology_name_prefix)