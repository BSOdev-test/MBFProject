import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import pipeline

def bug_fixing(buggycode):
     
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

    buggyCode = buggycode
    
    step1 = [
        {
            "role": "system",
            "content": "You are a chatbot who can help code.\n"
        },
        {
            "role": "user",
            "content": "You will be given an input. The input, after the line 'Buggy code:', contains a snippet of buggy Python code with a bug on a single line, and maybe unit test information. Given the input, analyze root cause and localize the faulty line by understanding the difference between actual output and expected output. After that, output 5 hypotheses about the fault's location and reason after the line 'Fault hypotheses:'. Rank these hypotheses by probability of success and use the information from the best ranked hypothesis to fix the received code.\n\nBuggy code: " + buggyCode + "\n\nFault hypothesis: \n"
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(step1, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    responseStep1 = outputs[0]["generated_text"]
    
    response = format_text(responseStep1)
    
    return response

def format_text(text):
    import re

    # Remover quebras de linha extras
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Adicionar quebras de linha antes de cada seção de comentário
    text = re.sub(r'(?<=\n)\s*#', '\n#', text)

    # Adicionar quebras de linha antes de blocos de código e tags de entrada/saída
    text = re.sub(r'(?<=\n)\s*(def|Buggy code:|###input###|###Actual output###|###Expected output###|Fault hypotheses:|5 hypotheses:|Ranking by probability of success:)', r'\n\1', text)

    # Ajustar as quebras de linha dentro das listas de linhas de saída
    text = re.sub(r'\[\s*\'', '[\'', text)
    text = re.sub(r'\'\s*\]', '\']', text)
    text = re.sub(r'\',\s*\'', '\', \'', text)

    # Corrigir indentação de código Python
    lines = text.split('\n')
    indent_level = 0
    formatted_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('def ') or stripped.startswith('class '):
            indent_level = 0
        if stripped:
            formatted_lines.append(' ' * indent_level + stripped)
        else:
            formatted_lines.append('')
        if stripped.endswith(':'):
            indent_level += 4
        elif stripped == '':
            indent_level = 0
    formatted_text = '\n'.join(formatted_lines)

    return formatted_text