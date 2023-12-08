# pruning ontologies with deeponto-0.5.4 (Yuan et al., 2022)
# setting maximum memory located to JVM [8g] to 32g 

# import ontology from deepOnto
from deeponto.onto import Ontology
from tqdm import tqdm

#load iris to keep
pruning_ratio="0.1"
fn_iris_to_keep = '../ontologies/SCTID_to_keep%s.txt' % pruning_ratio

with open(fn_iris_to_keep,encoding='utf-8') as f_content:
    lst_iris_to_keep = f_content.readlines()
set_iris_to_keep = set([iri.strip() for iri in lst_iris_to_keep]) # list to set - make the loop faster
print('num-classes-to-keep:',len(set_iris_to_keep))

#load ontology
fn_ontology = "../ontologies/SNOMEDCT-US-20170301-new.owl" 
#fn_ontology = "../ontologies/SNOMEDCT-US-20170301-new.pruned0.2.owl"
onto = Ontology(fn_ontology)

print('num-classes-all:',len(onto.owl_classes))
#print(onto.owl_classes)

lst_iris_to_remove = []
for SCTID in tqdm(onto.owl_classes):
    if not SCTID in set_iris_to_keep:
        lst_iris_to_remove.append(SCTID)

print('num-classes-to-remove:',len(lst_iris_to_remove))
print('start pruning')
onto.apply_pruning(lst_iris_to_remove)
onto.save_onto("../ontologies/SNOMEDCT-US-20170301-new.pruned%s-new.owl" % pruning_ratio)  # save the pruned ontology locally
