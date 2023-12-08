from transformers import GPT2Tokenizer
import pandas as pd

data_split = 'valid' # valid / test
snomed_subset='Disease' # Disease, CPP
top_k_base_value = 300
top_k_value = 300
filter_by_degree = False
prompts_fn = "../../models/biencoder/mm+%s2017AA-tl-sapbert-NIL-bs16/top%d_candidates/%s-NIL-top%d-preds%s-prompts-by-edges.csv" % (snomed_subset,top_k_base_value,data_split,top_k_value, "-degree-1" if filter_by_degree else "")

# Initialize GPT-2 tokenizer (GPT-3.5 uses a similar tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to calculate the number of sub-tokens for a given text
def count_subtokens(text):
    return len(tokenizer.tokenize(text))

prompts_df = pd.read_csv(prompts_fn,index_col=0)

# Applying the function to the "prompt" column
sub_token_counts = prompts_df['prompt'].apply(count_subtokens)

min_sub_token_length = sub_token_counts.min()
max_sub_token_length = sub_token_counts.max()

print(min_sub_token_length, max_sub_token_length)

'''
For Disease top-50:
    ~/BLINKout+/blink/prompting$ python count_sub_tokens.py
    1675 2610
    ~/BLINKout+/blink/prompting$ python count_sub_tokens.py
    1821 2629
For CPP top-50:
    ~/BLINKout+/blink/prompting$ python count_sub_tokens.py
    1556 2725
    ~/BLINKout+/blink/prompting$ python count_sub_tokens.py
    1634 3014
Disease valid top-300:
    ~/BLINKout+/blink/prompting$ python count_sub_tokens.py - 
    8863 12922
'''