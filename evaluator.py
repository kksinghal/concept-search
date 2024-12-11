import numpy as np
import sandbox
from transformation_to_vec import TransformationToVec
from hyperparameters import TransformationToVec_MODEL_DIR, SCORING_TYPE
from sentence_transformers import SentenceTransformer

np.random.seed(10)

transformation_to_vec = TransformationToVec(model_dir=TransformationToVec_MODEL_DIR)

"""
Parameters:
    f: function, program to evaluate
    inputs: List[2d list], input to program p
    outputs: List[2d list], true output of transformation

Returns: float in range[0,1], average of similarities between the transformations from input to true output and realised output
"""
def evaluate_similarity(f, inputs, outputs):
    if SCORING_TYPE == "neural":
        return neural_distance(f, inputs, outputs)
    elif SCORING_TYPE == "hamming_distance":
        return hamming_distance(f, inputs, outputs)


def neural_distance(f, inputs, outputs):
    assert len(inputs) == len(outputs)
    num_samples = len(inputs)
    total_score = 0
    
    realised_outputs = []
    for i in range(num_samples):
        syntactical_correctness, realised_output = sandbox.run(f, inputs[i])
        if not syntactical_correctness: return np.inf
        realised_outputs.append(realised_output)
        
    true_transformation_embedding = transformation_to_vec.get(inputs, outputs)
    pred_transformation_embedding = transformation_to_vec.get(inputs, realised_outputs)

    return np.sqrt(np.square(np.array(true_transformation_embedding) - np.array(pred_transformation_embedding)).sum())


model = SentenceTransformer("fintuned_sentence_transformer/checkpoint-1500", similarity_fn_name="euclidean")#("all-mpnet-base-v2")
print(model.similarity_fn_name)
def get_text_embedding(text):
    return model.encode(text)

def embedding_distance(h_embedding, goal_h_embedding):
    print("embedding ds:", model.similarity_fn_name)
    return -model.similarity(h_embedding, goal_h_embedding).squeeze()

def hypothesis_distance(h, goal_h):
    goal_h_embedding = model.encode(goal_h)
    h_embedding = model.encode(h)
    print(model.similarity_fn_name)
    return -model.similarity(goal_h_embedding, h_embedding).squeeze()


def get_program_hash(f, inputs):
    num_samples = len(inputs)
    realised_outputs = []
    for i in range(num_samples):
        syntactical_correctness, realised_output = sandbox.run(f, inputs[i])
        if not syntactical_correctness: return
        realised_outputs.append(realised_output)

    return tuple(transformation_to_vec.get(inputs, realised_outputs).tolist())

"""
Parameters:
    inp: List[2d list], input
    true_output: List[2d list]
    realised_output: List[2d list]

Returns: float in range[0,1], similarity between the transformation from input to true output and realised output
"""
def hamming_distance(f, inputs, outputs):
    assert len(inputs) == len(outputs)
    num_samples = len(inputs)
    total_score = 0

    realised_outputs = []
    for i in range(num_samples):
        syntactical_correctness, realised_output = sandbox.run(f, inputs[i])
        if not syntactical_correctness: return np.inf
        realised_outputs.append(realised_output)

    for o1, o2 in zip(outputs, realised_outputs):
        d = 0
        for i in range(min(len(o1), len(o2))):
            for j in range(min(len(o1[i]), len(o2[i]))):
                if o1[i][j] != o2[i][j]: d+=1
        
        total_score += max(len(o1),len(o2)) * max(len(o1[0]),len(o2[0])) - min(len(o1),len(o2)) * min(len(o1[0]),len(o2[0])) + d
    return total_score/len(inputs)


def pixelwise_comparison_score(f, inputs, outputs):
    assert len(inputs) == len(outputs)
    num_samples = len(inputs)
    total_score = 0
    for i in range(num_samples):
        _, realised_output = sandbox.run(f, inputs[i])
        total_score += compare_grid(outputs[i], realised_output)

    return total_score/num_samples


def compare_grid(grid1, grid2):
    if len(grid1) != len(grid2):
        return False
    
    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2) or any(elem1 != elem2 for elem1, elem2 in zip(row1, row2)):
            return False
    
    return True


def syntactical_correctness(f, inputs, outputs, return_error=False):
    assert len(inputs) == len(outputs)
    num_samples = len(inputs)
    for i in range(num_samples):
        correct, out = sandbox.run(f, inputs[i])
        if not correct: 
            print("Syntactically wrong")
            if return_error: return False, out
            else: return False
    
    if np.array(out).shape != np.array(outputs[-1]).shape:
        return True, "The output grid has incorrect shape"
    
    if return_error: return True, None
    else: return True