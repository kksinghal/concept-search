from time import sleep
from helper import string_to_function, get_model
import google.generativeai as genai
import random
import re

keys = [] #LIST OF KEYS
random.shuffle(keys)

class LLM:
    def __init__(self, temperature, model_name) -> None:
        self.models = []
        for key in keys:
            model, dsl = get_model(key, temperature, model_name)
            self.models.append(model)
            self.dsl_file = dsl
        

    def send_message(self, prompt, generate_code=True, key_index=0, iter=0):
        dsl_file = self.dsl_file
        while True:
            try:
                model = self.models[key_index%len(keys)]

                if generate_code:
                    response = model.generate_content([dsl_file, prompt])

                    string_res = response.text
                    if string_res.find('```python') == -1 or string_res.find('def') == -1: return self.send_message("Python code not found in output,\n" + prompt, True, key_index+1, iter+1)
                    fs = []
                    for m in re.finditer('```python', string_res):

                        start_idx = m.start() + 9
                        end_idx = string_res[start_idx:].find('```') + start_idx
                        code = string_res[start_idx:end_idx]

                        f, error = string_to_function(code)
                        fs.append(f)
                        if error is not None: 
                            if iter == 10: return string_res, [] # stop iteration 
                            return self.send_message(error + "\n" + prompt, True, key_index+1, iter+1)
                    
                    return string_res, fs
                else:
                    response = model.generate_content(prompt)
                    string_res = response.text
                
                return string_res
            except Exception as e:
                print(e)
                sleep(5)
                key_index+=1

    # def reset_session(self):
    #     self.chat_session = self.model.start_chat(history=[])
