import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import pipeline, AutoTokenizer

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("text-generation", model_path, torch_dtype=torch.bfloat16, device_map= "auto")

def bug_fixing(buggycode):
    analyses = phase1(buggycode)
    response = phase2(buggycode, analyses)
    return response
def phase1(buggycode):
    
    prompt = "Given a code snippet with an error, analyze and classify the type of error using contextual information. Error types can be syntax, runtime, logical, type, or compilation errors. Provide a concise classification and explanation. Code:\n" + buggycode
    
    chat = [
        {
            "role": "system",
            "content": "You are a debugging assistant.\n"
        },
        {
            "role": "user",
            "content": prompt
        },
    ]
    
    tokenized_chat = pipe.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    outputs = pipe(tokenized_chat, max_new_tokens=250, do_sample=True, temperature=0.75, top_k=10, top_p=0.95, num_return_sequences=5)
    
    analyses = []
    
    for i in outputs:
        output = format_text(i["generated_text"])
        output = extract_text_after_tag(output)
        analyses.append(output)
        
    return analyses
def phase2 (buggycode, analyses):
    
    prompt2 = "Given a code snippet with an error, you will be provided with five analyses that describe potential types of errors in the code. Your objective is to evaluate these analyses and select the one that most accurately identifies the error. Please ensure your choice is based on logical reasoning and programming principles.\nCode:\n" + buggycode + "\n" + "_" * 50 +"\nAnalyses:\n1: " + analyses[0] + "\n" + "_" * 50 + "\n2: " + analyses[1] + "\n" + "_" * 50 + "\n3: " + analyses[2] + "\n" + "_" * 50 + "\n4: " + analyses[3] + "\n" + "_" * 50 + "\n5: " + analyses[4]
    
    chat2 = [
        {
            "role": "system",
            "content": prompt2
        },
    ]
    
    tokenized_chat2 = pipe.tokenizer.apply_chat_template(chat2, tokenize=False, add_generation_prompt=False)
    outputs2 = pipe(tokenized_chat2, max_new_tokens=250, do_sample=True, temperature=0.75, top_k=10, top_p=0.95)
    response = format_text(outputs2[0]["generated_text"])
    
    # response = extract_text_after_tag(response)
    
    return response
def extract_text_after_tag(text):
 
    tag = '<|assistant|>'

    tag_index = text.find(tag)
    
    if tag_index != -1:
        return text[tag_index + len(tag):].strip()
    else:
        return ""  # Return an empty string if the tag is not found
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
def remove_text(text1, text2):
    if text1 in text2:
        print("Removed")
        return text2.replace(text1, '', 1).strip()  # Replace the first occurrence of text1 and remove leading/trailing spaces
    else:
        print("Not found")
        return text2  # Return text2 unchanged if text1 is not found