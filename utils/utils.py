import json
import time
import traceback

import re

import litellm

MODEL_NAME_TO_ID = {
    'chatgpt': 'gpt-3.5-turbo-0125',
    'gpt-4o': 'gpt-4o',
    # Claude models (using Bedrock inference profiles for newer models)
    'claude-v3-opus': 'us.anthropic.claude-3-opus-20240229-v1:0',
    'claude-v3-sonnet': 'us.anthropic.claude-3-sonnet-20240229-v1:0',
    'claude-v3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
    'claude-v3.5-sonnet': 'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
    'claude-v3.5-sonnet-v2': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    'claude-v3.5-haiku': 'anthropic.claude-3-5-haiku-20241022-v1:0',
    'claude-v3.7-sonnet': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    'claude-v4-sonnet': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
    'claude-v4-opus': 'us.anthropic.claude-opus-4-20250514-v1:0',
    'claude-v4.1-opus': 'us.anthropic.claude-opus-4-1-20250805-v1:0',
    'claude-v4.5-sonnet': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    'claude-v4.5-haiku': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    # Llama models
    'llama2-13b-chat': 'meta.llama2-13b-chat-v1',
    'llama2-70b-chat': 'meta.llama2-70b-chat-v1',
    'llama3-8b-instruct': 'meta.llama3-8b-instruct-v1:0',
    'llama3-70b-instruct': 'meta.llama3-70b-instruct-v1:0',
    'llama3.1-8b-instruct': 'meta.llama3-1-8b-instruct-v1:0',
    'llama3.1-70b-instruct': 'meta.llama3-1-70b-instruct-v1:0',
    'llama3.1-405b-instruct': 'meta.llama3-1-405b-instruct-v1:0',
    'llama3.2-11b-instruct': 'meta.llama3-2-11b-instruct-v1:0',
    'llama3.2-90b-instruct': 'meta.llama3-2-90b-instruct-v1:0',
    'llama3.3-70b-instruct': 'meta.llama3-3-70b-instruct-v1:0',
    'mistral-7b': 'mistral.mistral-7b-instruct-v0:2',
    'mixtral-8x7b': 'mistral.mixtral-8x7b-instruct-v0:1',
    'gemini-1.0-pro': 'gemini-1.0-pro-latest',
    'gemini-1.5-pro': 'gemini-1.5-pro-latest',
    'vicuna-7b-v1.5': 'vicuna-7b-v1.5',
    'vicuna-13b-v1.5': 'vicuna-13b-v1.5'
}

def get_litellm_response(prompt, model="claude-v3-sonnet", sleep_when_reaching_limit=10, return_type="full", **kwargs):
    """ Prompting LLMs to get response. Using the litellm implementation.
    Supported parameters in the kwargs:
        temperature, top_p,
        max_completion_tokens, max_tokens,
        stop, n
    
    return_type: {"text_only", "full"}
    """
    if isinstance(prompt, str):
        messages = [{"content": prompt,"role": "user"}]
    else:
        messages = prompt

    retry = 0
    while True:
        try:
            response = litellm.completion(
                model=MODEL_NAME_TO_ID.get(model, model),
                messages=messages,
                **kwargs
            )
            retry = 0
            break
        except litellm.RateLimitError as e:
            print(f"[RETRY #{retry} (wait for {sleep_when_reaching_limit})]", e)
            retry += 1
            time.sleep(sleep_when_reaching_limit)

    if return_type == "full":
        return response
    elif return_type == "text_only":
        return response.choices[0].message.content
    else:
        raise NotImplementedError("Unknown return_type given: "+return_type)


def get_llm_response(prompt, model, **kwargs):
    if "gpt" in model:
        kwargs.update(load_json('custom_api.json'))
    return get_litellm_response(prompt, model, **kwargs)

def load_json(dir_path, is_jsonl=False):
    if dir_path.endswith('.json') and (not is_jsonl):
        return json.load(open(dir_path, 'r'))
    elif dir_path.endswith('.jsonl') or is_jsonl:
        return [json.loads(line) for line in open(dir_path, 'r')]
    else:
        raise NotImplementedError("load_json not supported param combination")

def save_json(data, dir_path):
    if dir_path.endswith('.json'):
        with open(dir_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif dir_path.endswith(".jsonl"):
        with open(dir_path, 'w') as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

def decode_json(json_str):
    json_str = json_str.strip('```JSON\n').strip('```json\n').strip('\n```')
    json_str = json_str.replace('\n', '').replace('False', 'false').replace('True', 'true')
    try:
        return json.loads(json_str)
    except:
        return None

def parse_json(string):
    res = re.findall(r"{[\s\S]*}", string)
    if res:
        return json.loads(res[0])

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in {func.__name__}: {e}")
            tb = traceback.format_exc()
            print(f"Traceback:\n{tb}")
            return None  
    return wrapper


def apply_decorator_to_all_methods(decorator):
    def class_decorator(cls):
        for attr in dir(cls):
            if callable(getattr(cls, attr)) and not attr.startswith("__"):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return class_decorator


from functools import wraps
def retry(max_attempts=5, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                response = func(*args, **kwargs)
                if response is not None:
                    return response
                attempt += 1
                print(f"Attempt {attempt}/{max_attempts} failed.")
                time.sleep(delay)
            return response
        return wrapper
    return decorator



import json
import re
import ast
from typing import Any, Dict, List, Optional, Union


def extract_json(text: str, multiple: bool = False) -> Union[List, Dict, None]:
    """Extract JSON from LLM text using regex patterns."""
    if not text:
        return [] if multiple else None
    
    results = []
    
    # Find JSON objects and arrays
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # Code blocks
        r'```(?:json)?\s*(\[.*?\])\s*```',  # Array code blocks
        r'\{(?:[^{}]|{[^{}]*})*\}',         # Standalone objects
        r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]'    # Standalone arrays
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            parsed = _parse_json(match.strip())
            if parsed is not None and parsed not in results:
                results.append(parsed)
    
    return results if multiple else (results[0] if results else None)


def _parse_json(text: str) -> Optional[Union[Dict, List]]:
    """Parse JSON with fallback strategies."""
    # Clean text
    text = re.sub(r'^```(?:json)?\s*|```$', '', text.strip(), flags=re.IGNORECASE)
    
    # Try standard JSON
    try:
        return json.loads(text)
    except:
        pass
    
    # Fix common issues and retry
    fixed = text
    fixed = re.sub(r"'([^']*?)'", r'"\1"', fixed)  # Single quotes
    fixed = re.sub(r'\b(True|False|None)\b', lambda m: {'True': 'true', 'False': 'false', 'None': 'null'}[m.group()], fixed)
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)  # Trailing commas
    
    try:
        return json.loads(fixed)
    except:
        pass
    
    # Try Python literal eval
    try:
        result = ast.literal_eval(text)
        return result if isinstance(result, (dict, list)) else None
    except:
        pass
    
    return None


def parse_llm_json(text: str, expected_keys: List[str] = None) -> Dict[str, Any]:
    """Parse LLM response with validation."""
    json_obj = extract_json(text)
    
    if json_obj is None:
        return {"success": False, "data": None, "error": "No JSON found"}
    
    if expected_keys and isinstance(json_obj, dict):
        missing = [key for key in expected_keys if key not in json_obj]
        if missing:
            return {"success": False, "data": json_obj, "error": f"Missing keys: {missing}"}
    
    return {"success": True, "data": json_obj, "error": None}