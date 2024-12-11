import inspect
import sandbox
import uuid
import re
import importlib
from evaluator import evaluate_similarity
import traceback
import google.generativeai as genai
from hyperparameters import NUM_SAMPLES_WITH_SAME_PROMPT

def matrix_to_string(matrix):
    string = ""
    for row in matrix:
        string += " ".join([str(elem) for elem in row]) + "\n"
    return string

def program_to_string(program):
    return inspect.getsource(program)

def get_program_length(program):
    program_str = program_to_string(program)
    return len(program_str.split("\n"))

def generate_prompt_for_funsearch(io, programs, program_similarity_scores, program_errors): # made syntax mistake in last program
    syntax_prompt = ""
    for program_error in program_errors:
        syntax_prompt += "The following program raises the error given below it\n"
        syntax_prompt += f"{program_error[0]}\n"
        syntax_prompt += f"{program_error[1]}\n"

    initial_context = "We are playing a game which involves transforming an input 2d grid of digits into an output grid of digits using custom programming language, given in Domain-specific Language (dsl.py) file. In general, digits form objects in 2D and the task is to write code of some spatial transformation of these objects to go from the input grid to the output grid using only the functions given in Domain Specific Language (DSL). All the information about the transformation is contained within the input-ouput pairs themselves, and your answer will only be correct if the output grid is exactly correct, so this is what I expect from you. I will begin by giving you a sequence of input grids of the transformation. Then, I provide you a few programs and the corresponding transformation in the same sequence as input. These programs sampled from my database that look closest to the required transformation and but the output generated through them do not match required outputs. Each program has a similarity distance score between the transformation using program and the true transformation. Also, each of the output grid has a similarity distance score between that transformation example and true example, which can help you to identify which examples you need to focus on more. These similarity distances are inversely proportional to closeness between the transformations, so lower score is preferred and is your goal to achieve. \n"

    input_seqs = "Input grids: \n"
    for idx, (i, _) in enumerate(io):
        input_seqs += f"Input_{idx}:\n" + matrix_to_string(i) + "\n\n"

    program_outputs = ""
    for p_idx, program in enumerate(programs):
        program_outputs += f"Program_{chr(65+p_idx)}: with similarity distance score {program_similarity_scores[p_idx]}\n"
        program_outputs += program_to_string(program) + "\n"

        for idx, (i, o) in enumerate(io):
            _, realised_out = sandbox.run(program, tuple(map(tuple, i)))
            program_outputs += f"Output_{idx} of Input_{idx} using Program_{chr(65+p_idx)}\n" + matrix_to_string(realised_out) + "\n"

    required_transformation = "You must provide the code that transforms the inputs to following outputs (drives the score to 0) using only functions given in DSL. Analyse the above programs and their corresponding transformations. Then, analyse the required transformations given below to propose a program.\n"
    required_transformation += "Program_to_be_generated: (Complete this program only using functions from previous programs for following input-output transformation): with score 0\n"
    for idx, (i, o) in enumerate(io):
        required_transformation += f"Output_{idx} of Input_{idx} using Program_to_be_generated:\n" + matrix_to_string(o) + "\n"


    conclusion = f"""
Analysis & Code Generation:
Transformation Analysis: Analyze given input-output pairs and precisely describe the transformation logic applied to the input to achieve the output.
Code Improvement (x5): Provide five distinct versions of a program ("Program_to_be_generated") in a specified DSL (Domain Specific Language). Each version must be inspired by, but improve upon, two existing (unprovided in this prompt) programs.
Include a detailed explanation of how the program achieves the desired transformation for each input-output pair.
Code Extraction: Format the output to allow easy extraction of the "Program_to_be_generated" code.
Similarity: Ensure all five code versions are different, not just minor variations. Each should use distinct logic or DSL function combinations.
Output Format:
Transformation Analysis: A section for each input-output pair describing the transformation logic.
Program Version 1-{NUM_SAMPLES_WITH_SAME_PROMPT}: For each version:
Code: The complete "Program_to_be_generated" code in the DSL.
Explanation: A breakdown of how the code transforms each input into the desired output.
Essentially, you're being asked to act as a code-generating AI that can understand data transformations, improve existing code, and provide diverse solutions in the domain specific language from dsl file.
    """

    return syntax_prompt+initial_context + input_seqs + program_outputs + required_transformation + conclusion


def generate_prompt_for_direct_inference(io, test_input): # made syntax mistake in last program
    prompt = "We are playing a game which involves transforming an input grid of digits into an output grid of digits. In general, digits form objects in 2D and the task is to perform some spatial transformation of these objects to go from the input grid to the output grid. All the information about the transformation is contained within the input pairs themselves, and your answer will only be correct if the output grid is exactly correct, so this is what I expect from you. I will begin by giving you several examples of input-output pairs. You will then be given a new input grid, and you must provide the corresponding output grid.\n"

    for idx, (i, o) in enumerate(io):
        prompt += f"Input {idx+1}:\n"
        prompt += f"{matrix_to_string(i)}:\n"
        prompt += f"Output {idx+1}:\n"
        prompt += f"{matrix_to_string(o)}:\n\n"

    prompt += f"Input: {matrix_to_string(test_input)}\n"
    prompt += "Output 6: (please provide the output grid only)"
    return prompt
 


generated_uuids = set()
def generate_alphabetic_uuid():
    """Generates an alphabetic UUID.

    Returns:
    A string representing an alphabetic UUID.
    """

    uuid_string = str(uuid.uuid4())
    uuid_string = uuid_string.replace("-", "")
    uuid_string = uuid_string.replace("0", "a")
    uuid_string = uuid_string.replace("1", "b")
    uuid_string = uuid_string.replace("2", "c")
    uuid_string = uuid_string.replace("3", "d")
    uuid_string = uuid_string.replace("4", "e")
    uuid_string = uuid_string.replace("5", "f")
    uuid_string = uuid_string.replace("6", "g")
    uuid_string = uuid_string.replace("7", "h")
    uuid_string = uuid_string.replace("8", "i")
    uuid_string = uuid_string.replace("9", "j")

    if uuid_string in generated_uuids:
        return generate_alphabetic_uuid()

    generated_uuids.add(uuid_string)
    return uuid_string


# Returns program, error
def string_to_function(program):
    new_uuid = generate_alphabetic_uuid()
    program = re.sub(r'def.*\(', f'def {new_uuid}(', program, 1)

    with open(f"./temp_files/{new_uuid}.py", "w") as f:
        f.write("from dsl import *\n" + program)
        f.close()
    try:
        module = importlib.import_module("temp_files."+new_uuid)
        program = getattr(module, new_uuid)
    except Exception as e:
        return program, traceback.format_exc()

    return program, None

    
def get_model(key, temperature, model_name):
    genai.configure(api_key=key)
    generation_config = {
        "temperature": temperature,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )
    
    dsl_file = genai.upload_file(path="./dsl.py",
                        display_name="Domain-specific Language")
    return model, dsl_file


def goal_hypotheses_generation_prompt(k, train_inputs, train_outputs, similar_programs, program_hypotheses):
    prompt = "We are playing a game which involves transforming an input 2d grid of digits into an output grid of digits. In general, digits form objects in 2D and the task is to write code of some spatial transformation of these objects to go from the input grid to the output grid. All the information about the transformation is contained within the input-ouput pairs themselves.\n"
    prompt = f"I will begin by providing you with {len(similar_programs)} examples of different transformations and their corresponding descriptions. Your task is to determine a description of the required transformation given at the end.\n"
    for i in range(len(similar_programs)):
        prompt += f"Example {i}\n"
        p = similar_programs[i]
        h = program_hypotheses[i]

        for idx, inp in enumerate(train_inputs):
            prompt += f"Input_{idx}:\n" + matrix_to_string(inp) + "\n\n"
            _, out = sandbox.run(p, tuple(map(tuple, inp)))
            prompt += f"Output_{idx}:\n" + matrix_to_string(out) + "\n\n"

        prompt += f"Required Description of input-output transformation example {i}\n"
        prompt += h + "\n\n"

    prompt += "Example whose description is to be generated"
    for idx, (i,o) in enumerate(zip(train_inputs, train_outputs)):
        prompt += f"Input_{idx}:\n" + matrix_to_string(i) + "\n\n"
        prompt += f"Output_{idx}:\n" + matrix_to_string(o) + "\n\n"
    
    prompt += f"I want you to sample {k} different description describing the transformation logic from the input grids to output grids for the last example."
    prompt += "Write a new description in each new line in just plain text without any indexing, so that I can split the response using new line to retrieve the descriptions."
    return prompt


def generate_hypothesis_prompt(library, program):
    prompt = ""

    for idx, (p, desc) in enumerate(library):
        prompt += f"Example {idx}:\n"
        prompt += program_to_string(p)
        prompt += "Description: " + desc

        prompt += "\n\n"

    icl = prompt

    prompt = f"Example:\n"
    prompt += program_to_string(program)

    prompt += "Description:\n"
    prompt += "Generate only the description for last example"

    return icl + prompt
    