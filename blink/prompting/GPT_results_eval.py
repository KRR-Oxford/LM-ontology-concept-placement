# evaluate the results generated by Llama_2_prompting.py

import pandas as pd
# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)
from tqdm import tqdm
import re

snomed_subset='CPP'
top_k_base_value = 10 #50 #20 # 500
top_k_value = 10 #50 #20 # 50
filter_by_degree = False #Trues
is_NIL = True

model_name = "gpt-3.5-turbo" #"gpt-3.5-turbo" / "gpt-4"

def extract_leading_numbers(s):
    match = re.match(r'^\d+', s)
    return match.group(0) if match else None

def extract_trailing_numbers(s):
    match = re.search(r'\d+$', s)
    return match.group(0) if match else None

def unique_ordered(lst):
    return list(dict.fromkeys(lst))

def extract_pred(pred_raw):
    '''
    get number str from gen_raw
    '''    
    if pred_raw.startswith("None"):
        pred_raw = ""
        return pred_raw

    #list_pred_raw = pred_raw.split('\n')
    list_pred_raw = re.split(r'\\n|\n', pred_raw)
    list_pred = []
    for pred_raw_ele in list_pred_raw:
        pred_raw_ele_list = pred_raw_ele.split(',') 
        for pred_raw_ele_ele in pred_raw_ele_list:
            pred_raw_ele_ele = pred_raw_ele_ele.strip()            
            #if '.' in pred_raw_ele:
            #    pred_raw_ele = pred_raw_ele[:pred_raw_ele.find('.')]
            #list_pred.append(pred_raw_ele)
            if not pred_raw_ele_ele.isnumeric():
                leading_num = extract_leading_numbers(pred_raw_ele_ele)
                trailing_num = extract_trailing_numbers(pred_raw_ele_ele)
                if leading_num != None:
                    list_pred.append(leading_num)
                if trailing_num != None:
                    list_pred.append(trailing_num)    
            else:
                list_pred.append(pred_raw_ele_ele)   
    list_pred = unique_ordered(list_pred)        
    pred_raw = ','.join(list_pred)
    print('pred_raw:',pred_raw)
    return pred_raw.strip()

def eval_pred(pred,gold):
    '''
    both pred and gold are in the form of number1,number2,number3...
    the numbers are option numbers for the edges (-1 can appear in gold where the candidate is not found during the candidate creation process)

    TODO - add number of pred, top 5, and top 10 metrics
    '''
    #pred_numbers = pred.split(',') # this also handles heading and trailing spaces
    pred_numbers = [num_str.strip() for num_str in pred.split(',') if num_str.strip() != '']
    gold_numbers = gold.split(',')

    # get top-1 metric
    pred_top_1 = pred_numbers[0]
    is_top_1_any_correct = (pred_top_1 in gold_numbers)
    is_top_1_all_correct = (pred_top_1 == gold_numbers)
    

    # get top-k metric (k>1)
    pred_top_3 = pred_numbers[:3]
    is_top_3_any_correct = False
    is_top_3_all_correct = True
    for gold_number in gold_numbers:
        is_gold_in_pred = (gold_number in pred_top_3)
        is_top_3_any_correct = is_top_3_any_correct or is_gold_in_pred
        is_top_3_all_correct = is_top_3_all_correct and is_gold_in_pred
        
    pred_top_5 = pred_numbers[:5]
    is_top_5_any_correct = False
    is_top_5_all_correct = True
    for gold_number in gold_numbers:
        is_gold_in_pred = (gold_number in pred_top_5)
        is_top_5_any_correct = is_top_5_any_correct or is_gold_in_pred
        is_top_5_all_correct = is_top_5_all_correct and is_gold_in_pred

    pred_top_10 = pred_numbers[:10]
    is_top_10_any_correct = False
    is_top_10_all_correct = True
    for gold_number in gold_numbers:
        is_gold_in_pred = (gold_number in pred_top_10)
        is_top_10_any_correct = is_top_10_any_correct or is_gold_in_pred
        is_top_10_all_correct = is_top_10_all_correct and is_gold_in_pred

    # get "all" metric
    is_any_correct = False
    is_all_correct = True
    for gold_number in gold_numbers:
        is_gold_in_pred = (gold_number in pred_numbers)
        is_any_correct = is_any_correct or is_gold_in_pred
        is_all_correct = is_all_correct and is_gold_in_pred
    return len(pred_numbers), is_top_1_any_correct, is_top_1_all_correct, is_top_3_any_correct, is_top_3_all_correct, is_top_5_any_correct, is_top_5_all_correct, is_top_10_any_correct, is_top_10_all_correct, is_any_correct, is_all_correct

data_splits = ['valid','test']
#data_splits = ['test']
for data_split_mark in data_splits:
    result_filename = "../../models/biencoder/mm+%s2017AA-tl-sapbert-NIL-bs16/top%d_candidates/%s%s-top%d-preds%s-prompts-by-edges-%s.csv" % (snomed_subset,top_k_base_value,data_split_mark,'-NIL' if is_NIL else '',top_k_value, "-degree-1" if filter_by_degree else "",model_name)

    result_df = pd.read_csv(result_filename,index_col=0)
    #print(result_df.head())
    
    result_df['extracted_preds'] = ""
    result_df['num_preds'] = ""
    result_df['is_top_1_any_correct'] = ""
    result_df['is_top_1_all_correct'] = ""
    result_df['is_top_3_any_correct'] = ""
    result_df['is_top_3_all_correct'] = ""
    result_df['is_top_5_any_correct'] = ""
    result_df['is_top_5_all_correct'] = ""
    result_df['is_top_10_any_correct'] = ""
    result_df['is_top_10_all_correct'] = ""
    result_df['is_any_correct'] = ""
    result_df['is_all_correct'] = ""
    for i, row in tqdm(result_df.iterrows(),total=len(result_df)):        
        gold = str(row["answer"]).strip()
        if gold == "-1":
            continue
        gen_raw = row[model_name]
        pred = extract_pred(gen_raw)
        if pred != '':
            #print(gold, pred)
            num_preds, is_top_1_any_correct, is_top_1_all_correct, is_top_3_any_correct, is_top_3_all_correct,is_top_5_any_correct, is_top_5_all_correct, is_top_10_any_correct, is_top_10_all_correct, is_any_correct, is_all_correct = eval_pred(pred,gold)
            result_df.at[i,'extracted_preds'] = pred
            result_df.at[i,'num_preds'] = num_preds
            result_df.at[i,'is_top_1_any_correct'] = is_top_1_any_correct
            result_df.at[i,'is_top_1_all_correct'] = is_top_1_all_correct
            result_df.at[i,'is_top_3_any_correct'] = is_top_3_any_correct
            result_df.at[i,'is_top_3_all_correct'] = is_top_3_all_correct
            result_df.at[i,'is_top_5_any_correct'] = is_top_5_any_correct
            result_df.at[i,'is_top_5_all_correct'] = is_top_5_all_correct
            result_df.at[i,'is_top_10_any_correct'] = is_top_10_any_correct
            result_df.at[i,'is_top_10_all_correct'] = is_top_10_all_correct
            result_df.at[i,'is_any_correct'] = is_any_correct
            result_df.at[i,'is_all_correct'] = is_all_correct

    # print metrics
    ave_num_preds = result_df['num_preds'].apply(pd.to_numeric, errors='coerce').mean()
    num_ments = len(result_df)
    if data_split_mark == 'test':
        num_ments = num_ments + 1
    print('num_ments:',num_ments)
    insertion_rate_top1_any = len(result_df[result_df['is_top_1_any_correct']==True])/num_ments
    insertion_rate_top1_all = len(result_df[result_df['is_top_1_all_correct']==True])/num_ments
    insertion_rate_top3_any = len(result_df[result_df['is_top_3_any_correct']==True])/num_ments
    insertion_rate_top3_all = len(result_df[result_df['is_top_3_all_correct']==True])/num_ments
    insertion_rate_top5_any = len(result_df[result_df['is_top_5_any_correct']==True])/num_ments
    insertion_rate_top5_all = len(result_df[result_df['is_top_5_all_correct']==True])/num_ments
    insertion_rate_top10_any = len(result_df[result_df['is_top_10_any_correct']==True])/num_ments
    insertion_rate_top10_all = len(result_df[result_df['is_top_10_all_correct']==True])/num_ments
    insertion_rate_any = len(result_df[result_df['is_any_correct']==True])/num_ments
    insertion_rate_all = len(result_df[result_df['is_all_correct']==True])/num_ments    
    print('ave_num_preds',ave_num_preds,
          '\ninsertion_rate_top1_any',insertion_rate_top1_any,
          '\ninsertion_rate_top1_all',insertion_rate_top1_all,
          '\ninsertion_rate_top3_any',insertion_rate_top3_any,
          '\ninsertion_rate_top3_all',insertion_rate_top3_all,'\ninsertion_rate_top5_any',insertion_rate_top5_any,
          '\ninsertion_rate_top5_all',insertion_rate_top5_all,
          '\ninsertion_rate_top10_any',insertion_rate_top10_any,
          '\ninsertion_rate_top10_all',insertion_rate_top10_all,
          '\ninsertion_rate_any',insertion_rate_any,
          '\ninsertion_rate_all',insertion_rate_all,
    )

    filename_to_save = result_filename[:len(result_filename)-4] + '-annotation.xlsx'
    result_df.to_excel(filename_to_save,index=True)
    print('file saved to:', filename_to_save)
