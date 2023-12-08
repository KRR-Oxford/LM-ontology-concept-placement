from preprocessing.onto_snomed_owl_util import calculate_wu_palmer_sim,calculate_complex_wu_palmer_sim

def edge_wp_sim(onto_taxo,pred_edge_tuple,gold_edge_tuple,dict_iri_to_snd=None,dict_iri_pair_to_lca=None):
    '''
    calculate the wu & palmer sim between two edges, as the average value of parents' wu & palmer sim and children's in a pair of edges (predicted and gold)
    NOTE: for NULL node, use its pred or gold parent to infer the depth and the lowest common ancestor with another node
    '''
    pred_parent, pred_child = pred_edge_tuple
    gold_parent, gold_child = gold_edge_tuple
    # for parents: resolve into "owl:Thing" for the THING node 
    if pred_parent.endswith("_THING"):
        pred_parent = "owl:Thing"
    if gold_parent.endswith("_THING"):
        gold_parent = "owl:Thing"
    # for children: if the child (either in pred or in gold) in the edge is a NULL node, then, using its parent (either in pred or in gold) to calculate wu & palmer sim.
    get_NULL_node_depth_iri_pred = False
    get_NULL_node_depth_iri_gold = False
    if pred_child.endswith("_NULL"):
        pred_child = pred_parent
        assert not pred_child.endswith("_NULL")
        #print('NULL detected: uses pred parent for wp sim',pred_child)
        get_NULL_node_depth_iri_pred = True
    if gold_child.endswith("_NULL"):
        gold_child = gold_parent
        assert not gold_child.endswith("_NULL")
        #print('NULL detected: uses gold parent for wp sim',gold_child)
        get_NULL_node_depth_iri_gold = True
    parent_wp_sim, dict_iri_to_snd, dict_iri_pair_to_lca = calculate_wu_palmer_sim(onto_taxo,pred_parent,gold_parent,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca)
    child_wp_sim, dict_iri_to_snd, dict_iri_pair_to_lca = calculate_wu_palmer_sim(onto_taxo,pred_child,gold_child,
                                             get_NULL_node_depth_iri1=get_NULL_node_depth_iri_pred,
                                             get_NULL_node_depth_iri2=get_NULL_node_depth_iri_gold,
                                             dict_iri_to_snd=dict_iri_to_snd,
                                             dict_iri_pair_to_lca=dict_iri_pair_to_lca)
    return (parent_wp_sim + child_wp_sim) / 2, dict_iri_to_snd, dict_iri_pair_to_lca
    
def pred_edge_wp_sim(onto_taxo,pred_edge_tuple,list_gold_edge_tuples,dict_iri_to_snd=None,dict_iri_pair_to_lca=None):
    '''
    calculate the edge-level wu & palmer sim for a predicted edge.
    For each predicted edge, calculate *the highest score* of the predicted edge to any of the gold edges, each score is the average value of parents' wu & palmer sim and children's in a pair of edges (predicted and gold). 
    '''
    #assert len(list_gold_edge_tuples) > 0 # there should be at least one (atomic) gold edge
    for ind, gold_edge_tuple in enumerate(list_gold_edge_tuples):
        pred_gold_wp_sim,dict_iri_to_snd,dict_iri_pair_to_lca = edge_wp_sim(onto_taxo,pred_edge_tuple,gold_edge_tuple,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca)
        if ind == 0:
            pred_edge_score = pred_gold_wp_sim
        else:
            pred_edge_score = max(pred_edge_score, pred_gold_wp_sim)
    return pred_edge_score,dict_iri_to_snd,dict_iri_pair_to_lca

def overall_edge_wp_sim(onto_taxo,list_pred_edge_tuples,list_gold_edge_tuples,dict_iri_to_snd=None,dict_iri_pair_to_lca=None):
    '''
    calculate the overall edge-level wu & palmer sim.
    For each predicted edge, calculate the highest score of the predicted edge to any of the gold edges, each score is the average value of parents' wu & palmer sim and children's in a pair of edges (predicted and gold). 
    The overall score is the min/max/average of score over all the predicted edges.
    '''
    for ind, pred_edge_tuple in enumerate(list_pred_edge_tuples):
        pred_edge_score,dict_iri_to_snd,dict_iri_pair_to_lca = pred_edge_wp_sim(onto_taxo,pred_edge_tuple,list_gold_edge_tuples,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca)
        if ind == 0:
            overall_pred_gold_wp_sim_ave = pred_edge_score
            overall_pred_gold_wp_sim_min = pred_edge_score
            overall_pred_gold_wp_sim_max = pred_edge_score
        else:
            overall_pred_gold_wp_sim_ave += pred_edge_score
            overall_pred_gold_wp_sim_min = min(overall_pred_gold_wp_sim_min,pred_edge_score)
            overall_pred_gold_wp_sim_max = max(overall_pred_gold_wp_sim_max,pred_edge_score)
    overall_pred_gold_wp_sim_ave = overall_pred_gold_wp_sim_ave / len(list_pred_edge_tuples)
    return overall_pred_gold_wp_sim_ave,overall_pred_gold_wp_sim_min,overall_pred_gold_wp_sim_max,dict_iri_to_snd,dict_iri_pair_to_lca

# def eval_overall_edge_wp_sim_at_k(onto_taxo,list_pred_edge_tuples,list_gold_edge_tuples,dict_iri_to_snd=None,dict_iri_pair_to_lca=None,mode='max',k=5):
#     '''
#     eval_overall_edge_wp_sim_at_k by getting top-k from list_pred_edge_tuples 
#     '''
#     return overall_edge_wp_sim(onto_taxo,list_pred_edge_tuples[:k],list_gold_edge_tuples,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca,mode=mode)

## metrics for the case having complex edges

def edge_wp_sim_w_comp(onto_taxo,dict_SCTID_onto_obj_prop,pred_edge_tuple,gold_edge_tuple,dict_iri_to_snd=None,dict_iri_pair_to_lca=None,iri_prefix=""):
    '''
    calculate the wu & palmer sim between two edges, as the average value of parents' wu & palmer sim and children's in a pair of edges (predicted and gold)
    NOTE: for NULL node, use its pred or gold parent to infer the depth and the lowest common ancestor with another node
    '''
    pred_parent, pred_child = pred_edge_tuple
    gold_parent, gold_child = gold_edge_tuple
    # for parents: resolve into "owl:Thing" for the THING node 
    if pred_parent.endswith("_THING"):
        pred_parent = "owl:Thing"
    if gold_parent.endswith("_THING"):
        gold_parent = "owl:Thing"
    # for children: if the child (either in pred or in gold) in the edge is a NULL node, then, using its parent (either in pred or in gold) to calculate wu & palmer sim.
    get_NULL_node_depth_iri_pred = False
    get_NULL_node_depth_iri_gold = False
    if pred_child.endswith("_NULL"):
        pred_child = pred_parent
        assert not pred_child.endswith("_NULL")
        #print('NULL detected: uses pred parent for wp sim',pred_child)
        get_NULL_node_depth_iri_pred = True
    if gold_child.endswith("_NULL"):
        gold_child = gold_parent
        assert not gold_child.endswith("_NULL")
        #print('NULL detected: uses gold parent for wp sim',gold_child)
        get_NULL_node_depth_iri_gold = True
    parent_wp_sim, dict_iri_to_snd, dict_iri_pair_to_lca = calculate_complex_wu_palmer_sim(onto_taxo,dict_SCTID_onto_obj_prop,pred_parent,gold_parent,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca,iri_prefix=iri_prefix)
    child_wp_sim, dict_iri_to_snd, dict_iri_pair_to_lca = calculate_complex_wu_palmer_sim(onto_taxo,dict_SCTID_onto_obj_prop,pred_child,gold_child,
                                             get_NULL_node_depth_iri1=get_NULL_node_depth_iri_pred,
                                             get_NULL_node_depth_iri2=get_NULL_node_depth_iri_gold,
                                             dict_iri_to_snd=dict_iri_to_snd,
                                             dict_iri_pair_to_lca=dict_iri_pair_to_lca,iri_prefix=iri_prefix)
    return (parent_wp_sim + child_wp_sim) / 2, dict_iri_to_snd, dict_iri_pair_to_lca
    
def pred_edge_wp_sim_w_comp(onto_taxo,dict_SCTID_onto_obj_prop,pred_edge_tuple,list_gold_edge_tuples,dict_iri_to_snd=None,dict_iri_pair_to_lca=None,iri_prefix=""):
    '''
    calculate the edge-level wu & palmer sim for a predicted edge.
    For each predicted edge, calculate *the highest score* of the predicted edge to any of the gold edges, each score is the average value of parents' wu & palmer sim and children's in a pair of edges (predicted and gold). 
    '''
    #assert len(list_gold_edge_tuples) > 0 # there should be at least one (atomic) gold edge
    for ind, gold_edge_tuple in enumerate(list_gold_edge_tuples):
        pred_gold_wp_sim,dict_iri_to_snd,dict_iri_pair_to_lca = edge_wp_sim_w_comp(onto_taxo,dict_SCTID_onto_obj_prop,pred_edge_tuple,gold_edge_tuple,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca,iri_prefix=iri_prefix)
        if ind == 0:
            pred_edge_score = pred_gold_wp_sim
        else:
            pred_edge_score = max(pred_edge_score, pred_gold_wp_sim)
    return pred_edge_score,dict_iri_to_snd,dict_iri_pair_to_lca

def overall_edge_wp_sim_w_comp(onto_taxo,dict_SCTID_onto_obj_prop,list_pred_edge_tuples,list_gold_edge_tuples,dict_iri_to_snd=None,dict_iri_pair_to_lca=None,iri_prefix=None):
    '''
    calculate the overall edge-level wu & palmer sim.
    For each predicted edge, calculate the highest score of the predicted edge to any of the gold edges, each score is the average value of parents' wu & palmer sim and children's in a pair of edges (predicted and gold). 
    The overall score is the min/max/average of score over all the predicted edges.
    '''
    for ind, pred_edge_tuple in enumerate(list_pred_edge_tuples):
        pred_edge_score,dict_iri_to_snd,dict_iri_pair_to_lca = pred_edge_wp_sim_w_comp(onto_taxo,dict_SCTID_onto_obj_prop,pred_edge_tuple,list_gold_edge_tuples,dict_iri_to_snd=dict_iri_to_snd,dict_iri_pair_to_lca=dict_iri_pair_to_lca,iri_prefix=iri_prefix)
        if ind == 0:
            overall_pred_gold_wp_sim_ave = pred_edge_score
            overall_pred_gold_wp_sim_min = pred_edge_score
            overall_pred_gold_wp_sim_max = pred_edge_score
        else:
            overall_pred_gold_wp_sim_ave += pred_edge_score
            overall_pred_gold_wp_sim_min = min(overall_pred_gold_wp_sim_min,pred_edge_score)
            overall_pred_gold_wp_sim_max = max(overall_pred_gold_wp_sim_max,pred_edge_score)
    overall_pred_gold_wp_sim_ave = overall_pred_gold_wp_sim_ave / len(list_pred_edge_tuples)
    return overall_pred_gold_wp_sim_ave,overall_pred_gold_wp_sim_min,overall_pred_gold_wp_sim_max,dict_iri_to_snd,dict_iri_pair_to_lca