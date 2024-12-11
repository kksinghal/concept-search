import evaluator
import uuid
import numpy as np
import uuid
from evaluator import get_program_hash, get_text_embedding
from helper import get_program_length, program_to_string
from llm import LLM
from hyperparameters import SCORING_TYPE

np.random.seed(10)

class Island:
    def __init__(self, initial_programs, hamming_scores, hypothesis_scores, goal_hypothesis, train_inputs, train_outputs):
        self.id = uuid.uuid4()

        self.goal_hypothesis = goal_hypothesis
        self.goal_hypothesis_embedding = get_text_embedding(goal_hypothesis)

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs

        keys = [uuid.uuid4() for _ in range(len(initial_programs))]
        self.programs = dict(zip(keys, initial_programs))
        self.program_hypothesis_scores = dict(zip(keys, hypothesis_scores))
        self.program_hamming_scores = dict(zip(keys, hamming_scores))
        self.program_hashs = set()
        [self.program_hashs.add(get_program_hash(p, self.train_inputs)) for p in self.programs.values()]

    def get_top_k_programs(self, k, probabilistic=True):
        if SCORING_TYPE == "hamming":
            program_scores = self.program_hamming_scores
        else:
            program_scores = self.program_hypothesis_scores

        probs = np.array(list(program_scores.values()))
        probs = np.exp(-probs) + 1e-10
        probs /= probs.sum()
        if probabilistic:
            print(program_scores, k, list(program_scores.keys()))
            program_keys = np.random.choice(list(program_scores.keys()), size=k, replace=False, p=probs)
        else:
            scores = []
            program_lengths = []
            program_keys = []
            for p_key, p_score in program_scores.items():
                p = self.programs[p_key]
                program_keys.append(p_key)
                program_lengths.append(get_program_length(p))
                scores.append(p_score)

            scores = np.asarray(scores)
            program_lengths = np.asarray(program_lengths)
            program_keys = np.asarray(program_keys)

            data = np.array(list(zip(scores, program_lengths)), dtype=[('score', "float64"), ('program_length', 'int64')])

            idx = np.argpartition(data, k, order=["score", "program_length"])[:k]
            idx = idx[np.argsort(scores[idx])]
            program_keys = program_keys[idx]

        return [self.programs[key] for key in program_keys], [self.program_hamming_scores[key] for key in program_keys], [self.program_hypothesis_scores[key] for key in program_keys]

    def add_program(self, program, hypothesis):
        p_hash = get_program_hash(program, self.train_inputs)
        if p_hash in self.program_hashs: return -1, -1 # check if program that produces same input/output already exists

        self.program_hashs.add(p_hash)
        hamming_score = evaluator.hamming_distance(program, self.train_inputs, self.train_outputs)
        hypothesis_embed = evaluator.get_text_embedding(hypothesis)
        hypothesis_score = evaluator.embedding_distance(hypothesis_embed, self.goal_hypothesis_embedding)

        new_uuid = uuid.uuid4()
        self.programs[new_uuid] = program
        self.program_hamming_scores[new_uuid] = hamming_score
        self.program_hypothesis_scores[new_uuid] = hypothesis_score
        return hamming_score, hypothesis_score

    def write_to_log(self, text):
        f = open(f"./logs/islands/{self.id}.txt", "w")
        f.write(text+'\n')
        f.close()
