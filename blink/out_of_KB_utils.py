# utils for out-of-KB inference for BLINK
# (i) based on prediction scores of bi-encoder and cross-encoder
# (ii) based on classification - (to do)
# author: Hang Dong

import numpy as np

# numpy based softmax for 1D or 2D matrices
# from https://www.tutorialexample.com/implement-softmax-function-in-numpy-numpy-tutorial/
def _softmax(x):
    x_1d = False
    if x.ndim == 1:
        x = np.expand_dims(x, axis = 0)
        x_1d = True
    x = x - np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    x = x / np.sum(x, axis=1, keepdims=True)
    if x_1d:
        x = np.squeeze(x)
    return x

# infer out-of-KB entities from BM25 results using a BM25 score threshold
def infer_out_KB_ent_BM25(scores, nns, NIL_ent_id,th_NIL_BM25=0.0):
    for ind, score in enumerate(scores):
        if np.all(score <= th_NIL_BM25):
            # shift by one to the right, and insert NIL to the first
            nns[ind][1:] = nns[ind][:-1]
            nns[ind][0] = np.dtype('int64').type(NIL_ent_id) # or just, nns[ind][0] = NIL_ent_id
            print('nns[ind][0]:',nns[ind][0])
        # the below may be commented out - to do later
        else:
            # still giving it a chance (for cross-encoder) - by setting the last ranked entity as NIL. (this ensures recall@k 100% for NIL entities. more importantly, this is for evaluation of NIL entities in the cross-encoder process - otherwise there is no labels). Put it into other words, if we do not add NIL here, then this won't be correctly labelled in the evaluation data for cross-encoder (but as -1 since NIL is not found in the candidates), then for cross-encoder evaluation tp and t will be less counted.
            nns[ind][-1] = np.dtype('int64').type(NIL_ent_id)
            #pass
    return nns

# set NIL as the last top-k candidate if NIL is not predicted for the mention
# input: (i) nns, biencoder predicted top-k indexes as a list of arrays of entity indexes (each list for a mention)
#        (ii) NIL_ent_id, entity id for the out-of-KB / NIL entities
def set_NIL_to_candidates(nns, NIL_ent_id):
    for ind, nn in enumerate(nns):
        if not NIL_ent_id in nn:
            nns[ind][-1] = np.dtype('int64').type(NIL_ent_id)
    return nns

# infer out-of-KB entities from biencoder scores and adapt the predicted indexes (nns) 
# setting th_NIL_bi_enc as 0 means to replace the last entity to NIL (if it is not predicted in top-k)
# input: (i) scores, biencoder predicted scores as a list of arrays of mention-entity sim scores (each list for a mention)
#        (ii) nns, biencoder predicted top-k indexes as a list of arrays of entity indexes (each list for a mention)
#        (iii) NIL_ent_id, entity id for the out-of-KB / NIL entities
#        (iv) th_NIL_bi_enc, an out-of-KB mention-entity similarity threshold
def infer_out_KB_ent_bi_enc(scores, nns, NIL_ent_id,th_NIL_bi_enc=0.1):
    p_scores = _softmax(np.array(scores))
    #print('p_scores:',p_scores.shape,p_scores[0],p_scores[0]>th_NIL_bi_enc)
    # adjust nns of all values in p_score is below th_NIL_bi_enc
    #label_id_NIL = wikipedia_id2local_id['CUI-less']
    #print('NIL_ent_id:',NIL_ent_id,type(NIL_ent_id))
    #print('nns:',len(nns),type(nns[0]),type(nns[0][0]))
    for ind, p_score in enumerate(p_scores):
        # # check whether NIL is in nn
        # if NIL_ent_id in nns[ind]:
        #     # go to the next mention if NIL is in the top-k predictions for this mention
        #     continue
        if np.all(p_score <= th_NIL_bi_enc):
            # shift by one to the right, and insert NIL to the first
            nns[ind][1:] = nns[ind][:-1]
            nns[ind][0] = np.dtype('int64').type(NIL_ent_id) # or just, nns[ind][0] = NIL_ent_id
            #print('nns[ind][0]:',nns[ind][0])
        # the below may be commented out - to do later
        else:
            # still giving it a chance (for cross-encoder) - by setting the last ranked entity as NIL. (this ensures recall@k 100% for NIL entities. more importantly, this is for evaluation of NIL entities in the cross-encoder process - otherwise there is no labels). Put it into other words, if we do not add NIL here, then this won't be correctly labelled in the evaluation data for cross-encoder (but as -1 since NIL is not found in the candidates), then for cross-encoder evaluation tp and t will be less counted.
            # setting th_NIL_bi_enc as a negative value is the same as simply putting NIL at the end of prediction
            nns[ind][-1] = np.dtype('int64').type(NIL_ent_id)
            #pass
    return nns

# infer out-of-KB entities from crossencoder scores and adapt the predicted indexes (ind_out) 
# input: (i) logits, crosscoder predicted scores as a list of arrays (each list for a mention)
#        (ii) ind_out, cross-encoder predicted indexes as an array (each is an index of the candidates from the bi-encoder)
#        (iii) nns_batch, biencoder predicted/ranked indexes as a list of arrays of entity indexes (each list for a mention in the batch)
#        (iv) NIL_ent_id, entity id for the out-of-KB / NIL entities
#        (v) th_NIL_cross_enc, an out-of-KB mention-entity similarity threshold for crossencoder
# how it works: assign the NIL to the cross-encoder results (i) if it is in the candidates (as inferred with th_NIL_bi_enc after the bi-encoder); (ii) if the softmax logits (without the NIL one) all have score below the threshold, th_NIL_cross_enc.
# output: (i) updated ind_out
#         (ii) boolean numpy array indicating weather elements in ind_out is now changed to NIL
def infer_out_KB_ent_cross_enc(logits, ind_out, nns_batch, NIL_ent_id, th_NIL_cross_enc=0.6):
    #p_scores = _softmax(logits) # to note that here NIL-score (only in the first element in the candidates) is also softmaxed and being compared to the threshold - but this is fine, if NIL-score < th_NIL, then the NIL will be assigned to the predicted index; if not 
    #print('p_scores in train_cross._infer_out_KB_ent_cross_enc():',p_scores)
    #for ind, p_score_vec in enumerate(p_scores):
    candidate_size = len(nns_batch[0])    
    np_arr_bool_changed_to_NIL = np.full(ind_out.shape,False,dtype=bool)
    for ind, nn in enumerate(nns_batch):
        #if NIL_ent_id in nns_batch[ind]: # if the NIL entity id is found in the nns_batch.
        ind_NIL_ent_2d = np.where(nn==NIL_ent_id) #nns_batch[ind]
        #print('nns_batch[ind]:',nn) #nns_batch[ind]
        #print('NIL_ent_id:',NIL_ent_id)
        #print('ind_NIL_ent_2d:',type(ind_NIL_ent_2d),type(ind_NIL_ent_2d[0]),ind_NIL_ent_2d)    
        #ind_NIL_ent_2d: <class 'tuple'> <class 'numpy.ndarray'> (array([0]),)
        #print('ind_NIL_ent_2d:',ind_NIL_ent_2d)

        NIL_ent_id_in_cross = -1
        assert ind_NIL_ent_2d[0].size > 0 # assert that NIL's entity id is found in generated candidates for all mentions.
        if ind_NIL_ent_2d[0].size > 0: # to see if the NIL entity id is found in the nns.
            NIL_ent_id_in_cross = ind_NIL_ent_2d[0][0] #list(nns_batch[ind]).index(NIL_ent_id) # get the (first) index of the NIL entity in nns_batch 
            #print('NIL_ent_id_in_cross:',NIL_ent_id_in_cross)
            
        if NIL_ent_id_in_cross == 0:
            # assign the NIL to the cross-encoder results if it was inferred with th_NIL_bi_enc after the bi-encoder. (as the first ranked candidate, so the position is 0)
            ind_out[ind] = NIL_ent_id_in_cross 
            np_arr_bool_changed_to_NIL[ind] = True               
        else:
            if ind_out[ind] == NIL_ent_id_in_cross:
                #print('already predicted as NIL')
                np_arr_bool_changed_to_NIL[ind] = True
            else:    
                #assert NIL_ent_id_in_cross == candidate_size - 1 # if NIL entity is not the first one in the list of the ranked candidates, then it should be the last one. This does not necessarily hold as the bi-encoder may predict NIL.
                #print('potential NIL at trail')
                # if NIL_ent_id_in_cross == candidate_size - 1 
                #     logits_w_o_NIL = logits[:,:-1] # remove the logit of the NIL entity in the list
                # else:
                
                # remove the "NIL" column in the logits
                col_idxs = list(range(logits.shape[1]))
                col_idxs.pop(NIL_ent_id_in_cross) #this removes NIL-corresponded element from the list
                logits_w_o_NIL = logits[:, col_idxs]

                p_scores = _softmax(logits_w_o_NIL)
                p_score_vec = p_scores[ind]
                if np.all(p_score_vec <= th_NIL_cross_enc):
                    #print('potential NIL entity confirmed in cross-encoder')
                    # assign the NIL to the cross-encoder results if all the predicted scores by the cross-encoder are below a certain threshold.
                    ind_out[ind] = NIL_ent_id_in_cross
                    np_arr_bool_changed_to_NIL[ind] = True
                    ## set the predicted index as NIL_ent_id_in_cross, which is index of NIL in the candidates selected by the bi-encoder
                    #NIL_ent_id_in_cross = label_ids[ind] # seulement pour les label_ids qui were previously filtered with NIL.
                    
            #         ind_out[ind] = NIL_ent_id_in_cross
                # else:
                #     #print('already predicted as NIL')
                #     if ind_out[ind] == NIL_ent_id_in_cross:
                #         np_arr_bool_changed_to_NIL[ind] = True
    return ind_out,np_arr_bool_changed_to_NIL

# infer out of KB entity based on NIL classification in cross-encoder
def infer_out_KB_ent_cross_enc_classify(logits_NIL,ind_out, nns_batch, NIL_ent_id):
    np_arr_bool_changed_to_NIL = np.full(ind_out.shape,False,dtype=bool)
    print('logits_NIL:',logits_NIL)
    NIL_ind_out = np.argmax(logits_NIL, axis=1)
    print('NIL_ind_out:',NIL_ind_out)
    for ind, nn in enumerate(nns_batch):
        ind_NIL_ent_2d = np.where(nn==NIL_ent_id) #nns_batch[ind]
        print('nns_batch[ind]:',nn) #nns_batch[ind]
        #print('NIL_ent_id:',NIL_ent_id)
        #print('ind_NIL_ent_2d:',type(ind_NIL_ent_2d),type(ind_NIL_ent_2d[0]),ind_NIL_ent_2d)    
        #ind_NIL_ent_2d: <class 'tuple'> <class 'numpy.ndarray'> (array([0]),)
        print('ind_NIL_ent_2d:',ind_NIL_ent_2d)

        NIL_ent_id_in_cross = -1
        #assert ind_NIL_ent_2d[0].size > 0 # assert that NIL's entity id is found in generated candidates for all mentions.
        
        if ind_NIL_ent_2d[0].size > 0: # to see if the NIL entity id is found in the nns.
            NIL_ent_id_in_cross = ind_NIL_ent_2d[0][0] #list(nns_batch[ind]).index(NIL_ent_id) # get the (first) index of the NIL entity in nns_batch 
            #print('NIL_ent_id_in_cross:',NIL_ent_id_in_cross)
        else:
            print('warning: NIL not in bi-encoder pred for item %d in the batch' % ind)
            continue

        if NIL_ind_out[ind] == 1:
            # inferred as NIL
            print('inferred as NIL by classification')
            ind_out[ind] = NIL_ent_id_in_cross 
            np_arr_bool_changed_to_NIL[ind] = True

    return ind_out,np_arr_bool_changed_to_NIL