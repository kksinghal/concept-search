from helper import *
from island import Island
from hyperparameters import NUM_ISLANDS, NUM_FUNCTIONS_IN_PROMPT, RESET_PROBLEM_ITERS, NUM_ISLAND_RESETS, IMPROVE_ITERS, SCORING_TYPE
from helper import goal_hypotheses_generation_prompt
import evaluator
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import threading
from llm import LLM
import pandas as pd

random.seed(10)
np.random.seed(10)

class NeuralSearch:
    def __init__(self) -> None:
        self.llm_1 = LLM(temperature=1, model_name="gemini-1.5-pro-latest")
        self.llm_0 = LLM(temperature=0, model_name="gemini-1.5-pro-latest")
        
        self.h_llm = LLM(temperature=0, model_name="gemini-1.5-flash") # hypothesis generation


    def search(self, task_name, train_inputs, train_outputs, test_inputs, test_outputs, program_db):
        if not os.path.exists(f"./logs/{task_name}/"):
            os.makedirs(f"./logs/{task_name}/")

        self.score_logs = pd.DataFrame(columns=['reset_island_iter', 'improve_iter', 'island', *[f'program_{i+1}_hamming_score' for i in range(NUM_FUNCTIONS_IN_PROMPT)], *[f'program_{i+1}_hypothesis_score' for i in range(NUM_FUNCTIONS_IN_PROMPT)], *[f'output_program_{i+1}_hamming_score' for i in range(NUM_SAMPLES_WITH_SAME_PROMPT)], *[f'output_program_{i+1}_hypothesis_score' for i in range(NUM_SAMPLES_WITH_SAME_PROMPT)]])
        
        final_program = ""
        for reset_island_iter in range(NUM_ISLAND_RESETS): # change problem according to best program discovered so for

            # We might have a solution already in the library
            for p, desc in program_db.library:
                ans = evaluator.pixelwise_comparison_score(p, train_inputs, train_outputs)

                if (ans == 1):
                    if evaluator.pixelwise_comparison_score(p, test_inputs, test_outputs) == 1:
                        final_program += f"\n {program_to_string(p)} \n"
                        print(f"Done, found the solution in existing solvers at reset_iteration {reset_island_iter}")
                        with open(f"./logs/{task_name}/final_ans.txt", "w") as f:
                            f.write(f"found the solution in existing solvers at reset_iteration {reset_island_iter}\n{final_program}")

                        self.score_logs.to_csv(f"./logs/{task_name}/score_logs", sep='\t')
                        return 1
                
            similar_programs_hypotheses, hamming_scores, neural_scores = program_db.retrieve_top_k_programs_using_nn(10, train_inputs, train_outputs) # in increasing order of similarity
            similar_programs = [p for (p, h) in similar_programs_hypotheses]
            program_hypotheses = [h for (p, h) in similar_programs_hypotheses]
            prompt = goal_hypotheses_generation_prompt(NUM_ISLANDS, train_inputs, train_outputs, similar_programs, program_hypotheses)
            islands_goal_hypotheses = self.generate_goal_hypotheses(prompt)
            with open(f"./logs/{task_name}/goal_hypotheses.txt", "a") as f:
                f.write(str(islands_goal_hypotheses))

            islands = []
            for i in range(NUM_ISLANDS):
                similar_programs, hamming_scores, hypothesis_scores = program_db.retrieve_top_k_programs_using_hypothesis(NUM_FUNCTIONS_IN_PROMPT, islands_goal_hypotheses[i], train_inputs, train_outputs) # in increasing order of similarity 

                island = Island(similar_programs, hamming_scores, hypothesis_scores, islands_goal_hypotheses[i], train_inputs, train_outputs)
                islands.append(island)

            
            program_errors = [[] for _ in range(NUM_ISLANDS)]
            for i in range(IMPROVE_ITERS):
                
                # reorder input-output examples
                self.reorder_examples(train_inputs, train_outputs)

                island_prompts, island_responses, island_fs = [None]*NUM_ISLANDS, [None]*NUM_ISLANDS, [None]*NUM_ISLANDS
                threads = []
                for island_idx, island in enumerate(islands):
                    sampled_programs, hamming_scores, hypothesis_scores = island.get_top_k_programs(NUM_FUNCTIONS_IN_PROMPT, probabilistic=True)

                    if SCORING_TYPE == "hamming":
                        similarity_scores = hamming_scores
                    else:
                        similarity_scores = hypothesis_scores
                    island_prompts[island_idx] = generate_prompt_for_funsearch(list(zip(train_inputs, train_outputs)), sampled_programs, similarity_scores, program_errors[island_idx])
                    program_errors[island_idx] = [] # reset error feedback after usage
                    print("Funsearch happing", len(island_prompts[island_idx]))
                    t = threading.Thread(target=self.call_llm_thread, args=(island_responses, island_fs, island_idx, island_prompts[island_idx], reset_island_iter*IMPROVE_ITERS+i*NUM_ISLANDS+island_idx))
                    t.start()
                    threads.append(t)

                [t.join() for t in threads]

                print("code generation complete")

                for island_idx, island in enumerate(islands):
                    response, fs = island_responses[island_idx], island_fs[island_idx]

                    if not os.path.exists(f"./logs/{task_name}/search/problem_iter{reset_island_iter}/iter_{i}/"):
                        os.makedirs(f"./logs/{task_name}/search/problem_iter{reset_island_iter}/iter_{i}/")

                    open(f"./logs/{task_name}/search/problem_iter{reset_island_iter}/iter_{i}/island_{island_idx}_prompt.txt", "w").write(island_prompts[island_idx])
                    open(f"./logs/{task_name}/search/problem_iter{reset_island_iter}/iter_{i}/island_{island_idx}_response.txt", "w").write(response)


                    score_log = {
                        'reset_island_iter': reset_island_iter, 
                        'improve_iter': i, 
                        'island': island_idx
                    }
                    for idx in range(NUM_FUNCTIONS_IN_PROMPT):
                        score_log[f'program_{idx+1}_hamming_score']=hamming_scores[idx]
                        score_log[f'program_{idx+1}_hypothesis_score']=hypothesis_scores[idx]


                    for idx, f in enumerate(fs):
                        # Next three functions could be done once, but keeping it separate for readability
                        syntax_correctness, error = evaluator.syntactical_correctness(f, train_inputs, train_outputs, return_error=True)
                        if not syntax_correctness:
                            program_errors[island_idx].append([program_to_string(f), error])
                            score_log[f'output_program_{idx+1}_hamming_score'] = np.inf
                            score_log[f'output_program_{idx+1}_hypothesis_score'] = np.inf

                        elif syntax_correctness:
                            h = self.generate_hypothesis(program_db, f)
                            hamming_score, hypothesis_score = island.add_program(f, h) #Workaround: write string code p to a python file and then import it again to get its definition
                            # otherwise, exec just creates the function, but we don't have its definition
                            score_log[f'output_program_{idx+1}_hamming_score'] = hamming_score
                            score_log[f'output_program_{idx+1}_hypothesis_score'] = hypothesis_score

                            if hamming_score == -1: continue # program already exists

                            with open(f"./logs/{task_name}/generated_program_scores.txt", "a") as file:
                                file.write(f"{str(hamming_score)}, {str(hypothesis_score)}, {h}:: {program_to_string(f)} \n")
                                file.close()

                            program_db.add_programs([[f, h]])

                            ans = evaluator.pixelwise_comparison_score(f, train_inputs, train_outputs)
                            # print(ans, program_to_string(f))
                            if (ans == 1):
                                if evaluator.pixelwise_comparison_score(f, test_inputs, test_outputs) == 1:
                                    final_program += f"\n {program_to_string(f)} \n"
                                    print(f"Done, found the solution in existing solvers at reset_iteration {reset_island_iter}, improve iteration {i}")
                                    with open(f"./logs/{task_name}/final_ans.txt", "w") as f:
                                        f.write(f"found the solution in existing solvers at reset_iteration {reset_island_iter}, improve iteration {i}\n{final_program}")

                                    self.score_logs.loc[len(self.score_logs)] = score_log
                                    self.score_logs.to_csv(f"./logs/{task_name}/score_logs", sep='\t')
                                    return 1
                    
                    self.score_logs.loc[len(self.score_logs)] = score_log


            if (reset_island_iter+1)%RESET_PROBLEM_ITERS == 0:
                # reset problem
                best_program_data = [island.get_top_k_programs(1, probabilistic=False) for island in islands] # (program, score)
                best_programs = [p[0][0] for p in best_program_data] 
                best_program_scores = [p[1][0] for p in best_program_data] 

                best_program_idx = np.argmax(best_program_scores)
                best_program = best_programs[best_program_idx]

                final_program += f"\n {program_to_string(best_program)} \n"

                new_train_inputs = []
                new_test_inputs = []
                for i in range(len(train_inputs)):
                    error_free, realised_output = sandbox.run(best_program, train_inputs[i])

                    new_train_inputs.append(realised_output)

                for i in range(len(test_inputs)):
                    _, realised_output = sandbox.run(best_program, test_inputs[i])
                    new_test_inputs.append(realised_output)

                train_inputs = new_train_inputs
                test_inputs = new_test_inputs

        with open(f"./logs/{task_name}/final_ans.txt", "w") as f:
            f.write("solution not found\n" + final_program)

        self.score_logs.to_csv(f"./logs/{task_name}/score_logs", sep='\t')
        return 0


    def reorder_examples(self, train_inputs, train_outputs):
        temp = list(zip(train_inputs, train_outputs))
        random.shuffle(temp)
        train_inputs, train_outputs = zip(*temp)
        train_inputs, train_outputs = list(train_inputs), list(train_outputs)
        return train_inputs, train_outputs
    
    def call_llm_thread(self, island_responses, island_fs, island_idx, prompt, key_index):
        island_responses[island_idx], island_fs[island_idx] = self.llm_1.send_message(prompt, generate_code=True, key_index=key_index) # string response, program f

    def generate_goal_hypotheses(self, prompt):
        hypotheses = self.llm_0.send_message(prompt, generate_code=False, key_index=len(prompt)).split("\n")
        return [h for h in hypotheses if bool(h) == True]

    def generate_hypothesis(self, program_db, program):
        prompt = generate_hypothesis_prompt(program_db.library, program)
        description = self.h_llm.send_message(prompt, generate_code=False)
        return description
