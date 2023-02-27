import transformers

import torch
from transformers import OpenAIGPTTokenizer, GPT2Tokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import faulthandler
faulthandler.enable()

from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
#model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
#tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
print("><>before model")
#model = AutoModelWithLMHead.from_pretrained('/Users/beeoladeji/Desktop/content/gpt-2/output')
#model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('/Users/beeoladeji/Desktop/content/gpt-2/output')
print("**after model")
#/Users/beeoladeji/miniconda3/pkgs/pydotplus-2.0.2-py_3/site-packages/pydotplus-2.0.2.dist-info/METADATA
def generate_answer(question):
    new_user_input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    #bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) 

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(new_user_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, temperature=0.6, repetition_penalty=1.3)

    preds = [ tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)]
    # pretty print last ouput tokens from bot
    #print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    return "".join(preds)



if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = generate_answer(sentence)
        print(resp)

#/Users/beeoladeji/Desktop/output
# import os
# import pandas as pd
# import openai 

# import openai
# openai.Completion.create(
#     model=FINE_TUNED_MODEL,
#     prompt=YOUR_PROMPT)



# def ask(question,chat_log = None):
#   if chat_log is None:
#     chat_log = start_chart_log
#   prompt = f'{chat_log}Human:{question}\nAI:'
#   print("prompt",prompt)
#   response = completion.create(
#       prompt = prompt,engine = "davinci",stop = ["\nHuman"],temperature = 0.9,
#       top_p =1,best_of=1,
#       max_tokens=150
#   )
#   answer = response.choices[0].text.strip()
#   return answer

#export OPENAI_API_KEY="<OPENAI_API_KEY>"
#openai tools fine_tunes.prepare_data -f <LOCAL_FILE>