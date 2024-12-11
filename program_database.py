import numpy as np
import evaluator
from tqdm import tqdm
from helper import program_to_string, get_program_length
from evaluator import get_text_embedding, embedding_distance
from hyperparameters import SCORING_TYPE

np.random.seed(10)

def write(string):
    with open("./logs/programdb_scores.txt", "a") as f:
        f.write(string)

########## DSL ###########
class ProgramDatabase:
    """
    Parameters:
        initial_library: List[List[function], string]
    """
    def __init__(self, initial_library):
        self.library = initial_library
        self.hypothesis_embeddings = get_text_embedding([desc for p, desc in self.library]).tolist()

    def add_programs(self, programs_hypotheses):
        self.library += programs_hypotheses
        self.hypothesis_embeddings += get_text_embedding([desc for p, desc in programs_hypotheses]).tolist()

    """
    Parameters:
        k: int, number of programs to sample
        inputs: List[2d list], input to transformation
        outputs: List[2d list], true output of transformation
    
    Returns: (List[function], List[np.float], List[np.float])
        top k programs in increasing order of similarity to given transformation, their similarity score
    """
    def retrieve_top_k_programs_using_nn(self, k, inputs, outputs):
        print("Finding most similar tasks...")
        hamming_scores = []
        neural_scores = []
        program_lengths = []
        for p, description in tqdm(self.library):
            hamming_score = evaluator.hamming_distance(p, inputs, outputs)
            neural_score = evaluator.neural_distance(p, inputs, outputs)
            write(f"{program_to_string(p)} {hamming_score}, {neural_score}")
            hamming_scores.append(hamming_score)
            neural_scores.append(neural_score)
            program_lengths.append(get_program_length(p))

        scores = np.asarray(neural_scores)

        hamming_scores = np.asarray(hamming_scores)
        neural_scores = np.asarray(neural_scores)
        
        program_lengths = np.asarray(program_lengths)

        data = np.array(list(zip(scores, program_lengths)), dtype=[('score', "float64"), ('program_length', 'int64')])

        idx = np.argpartition(data, k, order=["score", "program_length"])[:k]
        idx = idx[np.argsort(scores[idx])]

        return [self.library[index] for index in idx], hamming_scores[idx], neural_scores[idx]
    

    """
    Parameters:
        k: int, number of programs to sample
        goal_hypothesis: string
        inputs: List[2d list], input to transformation
        outputs: List[2d list], true output of transformation
    
    Returns: (List[function], List[np.float], List[np.float])
        top k programs in increasing order of similarity to given transformation, their similarity score
    """
    def retrieve_top_k_programs_using_hypothesis(self, k, goal_hypothesis, inputs, outputs):
        print("Finding most similar tasks...")

        goal_h_embedding = get_text_embedding(goal_hypothesis)

        hypothesis_scores = embedding_distance(self.hypothesis_embeddings, goal_h_embedding).tolist()
        print("Hellooooo", hypothesis_scores)

        hamming_scores = []

        program_lengths = []
        for idx, (p, description) in tqdm(enumerate(self.library)):
            hamming_score = evaluator.hamming_distance(p, inputs, outputs)
            # hypothesis_score = evaluator.hypothesis_distance(description, goal_hypothesis)
            write(f"{program_to_string(p)} {hamming_score}, {hypothesis_scores[idx]}")
            hamming_scores.append(hamming_score)
            program_lengths.append(get_program_length(p))

        scores = np.asarray(hypothesis_scores)

        hamming_scores = np.asarray(hamming_scores)
        hypothesis_scores = np.asarray(hypothesis_scores)
        
        program_lengths = np.asarray(program_lengths)

        data = np.array(list(zip(scores, program_lengths)), dtype=[('score', "float64"), ('program_length', 'int64')])

        idx = np.argpartition(data, k, order=["score", "program_length"])[:k]
        idx = idx[np.argsort(scores[idx])]

        return [self.library[index][0] for index in idx], hamming_scores[idx], hypothesis_scores[idx]