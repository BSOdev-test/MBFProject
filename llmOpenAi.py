import os
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def bug_fixing(buggycode):
    analysis = phase1(buggycode)
    description = phase2(buggycode)
    result = phase3(buggycode, description, analysis)
    return result

def phase1(buggycode):
    prompt = "Given a code snippet with an error, analyze and classify the type of error using contextual information. Error types can be syntax, runtime, logical, type, or compilation errors. Provide a concise classification and explanation.\nCode:\n" + buggycode

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0
    )
    
    analysis = response.choices[0].message.content
    print("_" *100 + "\n" + analysis + "\n" + "_" *100)
    return analysis

def phase2(buggycode):
    prompt = "Given a code snippet, analyze and bring a step by step description of the code. \nCode:\n" + buggycode

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0
    )
    
    description = response.choices[0].message.content
    print("_" *100 + "\n" + description + "\n" + "_" *100)
    return description

def phase3 (buggycode, description, analysis):
    
    prompt2 = "Given a code snippet with an error, you will be provided with a description of the code and a analysis that describe potential types of errors in the code. Your objective is to evaluate the analysis and the description to explain why and where the code has an error. Please ensure your choice is based on logical reasoning and programming principles.\nCode:\n" + buggycode + "\nDescription:\n" + description + "\nAnalysis:\n" + analysis
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt2,
            }
        ],
        model=model_name,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0
    )
    
    result = response.choices[0].message.content
    
    return result