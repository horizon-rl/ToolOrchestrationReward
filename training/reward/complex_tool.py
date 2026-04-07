# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple

# Import BFCL AST checker for function call validation
import sys
import os
# Add BFCL to path if not already there
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.getenv("REPO_ROOT", os.path.abspath(os.path.join(_script_dir, "..", "..")))
bfcl_path = os.path.join(_repo_root, "BFCL")
if os.path.exists(bfcl_path) and bfcl_path not in sys.path:
    sys.path.insert(0, bfcl_path)

try:
    from bfcl_eval.eval_checker.ast_eval.ast_checker import simple_function_checker
    from bfcl_eval.constants.enums import Language
except ImportError as e:
    raise ImportError(
        f"Failed to import BFCL AST checker. This is required for R_atomic_validity reward computation. "
        f"Make sure BFCL is properly installed at {bfcl_path}. Original error: {e}"
    ) from e


def types_compatible(v1: Any, v2: Any) -> bool:
    """
    Check if two values have compatible types for validation.

    Allows int/float interchangeability since JSON doesn't distinguish
    between them (e.g., 2 and 2.0 are both valid representations).

    Args:
        v1: First value
        v2: Second value

    Returns:
        True if types are compatible, False otherwise
    """
    # Allow int/float interchangeability
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return True
    # Strict type check for all other types
    return type(v1) == type(v2)


def safe_get_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely extract and normalize arguments from a tool call.

    Handles cases where the model generates malformed tool calls with
    arguments as a JSON string instead of a dict. Returns empty dict
    for malformed cases, which will result in zero reward.

    Args:
        tool_call: Tool call dict with 'arguments' field

    Returns:
        Dict of arguments, or empty dict if malformed
    """
    # Validate tool_call is a dict
    if not isinstance(tool_call, dict):
        logger.warning(
            f"Tool call is {type(tool_call).__name__}, expected dict. "
            f"Value: {tool_call}. This will receive zero reward."
        )
        return {}

    arguments = tool_call.get('arguments', {})

    if isinstance(arguments, dict):
        return arguments
    elif isinstance(arguments, str):
        # Model generated malformed format: arguments as string instead of dict
        logger.warning(
            f"Malformed tool call detected: arguments is a string instead of dict. "
            f"Tool: {tool_call.get('name', 'unknown')}, arguments: {arguments[:100]}... "
            f"This will receive zero reward."
        )
        # Try to parse as JSON in case it's valid JSON string
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                logger.warning(f"Successfully recovered arguments by parsing JSON string")
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {}
    else:
        logger.warning(
            f"Malformed tool call detected: arguments has unexpected type {type(arguments).__name__}. "
            f"Tool: {tool_call.get('name', 'unknown')}. This will receive zero reward."
        )
        return {}


def extract_tool_calls_and_responses(text: str) -> Dict[str, Any]:
    """
    Extract tool calls and their corresponding responses from function output text.
    
    Args:
        text (str): The function output string containing tool calls and responses
        
    Returns:
        Dict containing:
        - 'tool_calls': List of parsed tool calls
        - 'tool_responses': List of parsed tool responses  
        - 'call_response_pairs': List of tuples pairing calls with their responses
        - 'unpaired_calls': List of tool calls without responses
        - 'unpaired_responses': List of tool responses without calls
    """
    
    # Extract tool calls - extract everything between tags, let JSON parser handle structure
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_calls = []

    for match in re.finditer(tool_call_pattern, text, re.DOTALL):
        json_str = match.group(1).strip()
        try:
            call_data = json.loads(json_str)
            tool_calls.append({
                'data': call_data,
                'raw': match.group(0),
                'position': match.start()
            })
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}")
            tool_calls.append({
                'data': None,
                'raw': match.group(0),
                'position': match.start(),
                'parse_error': str(e)
            })

    # Extract tool responses - extract everything between tags, let JSON parser handle structure
    tool_response_pattern = r'<tool_response>(.*?)</tool_response>'
    tool_responses = []

    for match in re.finditer(tool_response_pattern, text, re.DOTALL):
        json_str = match.group(1).strip()
        try:
            response_data = json.loads(json_str)
            tool_responses.append({
                'data': response_data,
                'raw': match.group(0),
                'position': match.start()
            })
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool response JSON: {e}")
            tool_responses.append({
                'data': None,
                'raw': match.group(0),
                'position': match.start(),
                'parse_error': str(e)
            })
    
    # Pair tool calls with their responses
    call_response_pairs = []
    unpaired_calls = tool_calls.copy()
    unpaired_responses = tool_responses.copy()
    used_responses = set()  # Track used responses to prevent duplicate pairing

    # For each tool call, find the next tool response after it
    for call in tool_calls:
        matching_response = None
        for response in tool_responses:
            # Check if response is after call AND hasn't been used yet
            if response['position'] > call['position'] and id(response) not in used_responses:
                matching_response = response
                used_responses.add(id(response))  # Mark as used
                break

        if matching_response:
            call_response_pairs.append((call, matching_response))
            if call in unpaired_calls:
                unpaired_calls.remove(call)
            if matching_response in unpaired_responses:
                unpaired_responses.remove(matching_response)
    
    return {
        'tool_calls': [call['data'] for call in tool_calls if call['data'] is not None],
        'tool_responses': [response['data'] for response in tool_responses if response['data'] is not None],
        'call_response_pairs': [(call['data'], response['data']) for call, response in call_response_pairs 
                               if call['data'] is not None and response['data'] is not None],
        'unpaired_calls': [call['data'] for call in unpaired_calls if call['data'] is not None],
        'unpaired_responses': [response['data'] for response in unpaired_responses if response['data'] is not None],
        'raw_calls': [call['raw'] for call in tool_calls],
        'raw_responses': [response['raw'] for response in tool_responses],
        'parse_errors': [item for item in tool_calls + tool_responses if 'parse_error' in item]
    }



# def compute_model_reward(solutions, ground_truth, is_matched, weight_mode='uniform'):
#     """
#     Compute step-level and global success rewards for model solutions vs ground truth.
    
#     Args:
#         solutions: List of function calls [call_1, call_2, ...]
#         ground_truth: List of steps [[step_1_call_1, step_1_call_2, ...], [step_2_...], ...]
#         is_matched: Function that takes (func_call, gt_func_call) and returns True if matched
#         weight_mode: 'uniform' for equal weights, 'dependency' for dependency-based weights
    
#     Returns:
#         dict: {
#             'step_rewards': [R_step_1, R_step_2, ...],
#             'step_weights': [weight_1, weight_2, ...],
#             'global_reward': R_success
#         }
#     """
#     step_rewards = []
#     step_weights = []
    
#     for step_idx, gt_step in enumerate(ground_truth):
#         # Step succeeds if ANY function call in the step is matched
#         R_step_i = 0.0
        
#         for gt_call in gt_step:
#             # Check if any solution call matches this ground truth call
#             for sol_call in solutions:
#                 if is_matched(sol_call, gt_call):
#                     R_step_i = 1.0
#                     break
#             if R_step_i == 1.0:  # Stop searching once we find a match
#                 break
                
#         step_rewards.append(R_step_i)
        
#         # Calculate step weight
#         if weight_mode == 'uniform':
#             weight = 1.0
#         elif weight_mode == 'dependency':
#             weight = len(gt_step)
#         else:
#             weight = 1.0
            
#         step_weights.append(weight)
    
#     # Calculate global success reward as weighted average
#     if sum(step_weights) > 0:
#         weighted_sum = sum(r * w for r, w in zip(step_rewards, step_weights))
#         R_success = weighted_sum / sum(step_weights)
#     else:
#         R_success = 0.0
    
#     return {
#         'step_rewards': step_rewards,
#         'step_weights': step_weights,
#         'global_reward': R_success
#     }

def convert_to_bfcl_format(tool_call: Dict[str, Any], ground_truth_trace: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
    """
    Convert our tool call and ground truth to BFCL's expected format.

    Args:
        tool_call: Model's tool call {'name': 'func_name', 'arguments': {...}}
        ground_truth_trace: Ground truth trace with function details

    Returns:
        Tuple of (func_description, model_output, possible_answer) in BFCL format
    """
    if 'traces' not in ground_truth_trace or not ground_truth_trace['traces']:
        return None, None, None

    # Use the first successful trace as the template
    gt_trace = ground_truth_trace['traces'][0]

    # Build function description (schema)
    func_description = {
        'name': gt_trace['function_name'],
        'parameters': {
            'type': 'dict',
            'required': list(gt_trace['arguments'].keys()),
            'properties': {}
        }
    }

    # Infer parameter types from ground truth arguments
    for param_name, param_value in gt_trace['arguments'].items():
        param_type = type(param_value).__name__

        # Handle array types specially (BFCL requires 'items' field)
        if param_type == 'list':
            param_schema = {'type': 'array'}

            # Infer items type from first element (if available)
            if isinstance(param_value, list) and len(param_value) > 0:
                first_item = param_value[0]
                item_type = type(first_item).__name__

                # Convert Python types to BFCL expected types
                if item_type == 'str':
                    item_type = 'string'
                elif item_type == 'int':
                    item_type = 'integer'
                elif item_type == 'bool':
                    item_type = 'boolean'
                elif item_type == 'dict':
                    item_type = 'dict'  # BFCL uses 'dict', not 'object'
                elif item_type == 'list':
                    item_type = 'array'

                param_schema['items'] = {'type': item_type}
            else:
                # Empty array - default to string items
                param_schema['items'] = {'type': 'string'}

            func_description['parameters']['properties'][param_name] = param_schema
            continue

        # Handle other types
        if param_type == 'str':
            param_type = 'string'
        elif param_type == 'int':
            param_type = 'integer'
        elif param_type == 'bool':
            param_type = 'boolean'
        elif param_type == 'dict':
            param_type = 'dict'  # BFCL uses 'dict', not 'object'

        func_description['parameters']['properties'][param_name] = {
            'type': param_type
        }

    # Convert model output to BFCL format
    # Use safe_get_arguments to handle malformed tool calls
    model_output = {
        gt_trace['function_name']: safe_get_arguments(tool_call)
    }

    # Build possible answer (ground truth values)
    # Use strict matching only - no variable wildcards
    # BFCL expects values to be lists, so wrap in list with single value
    possible_answer = {
        gt_trace['function_name']: {}
    }
    for param_name, param_value in gt_trace['arguments'].items():
        # Strict value matching for Level 3 validation
        # Wrap in list (BFCL requirement) but only include exact value (no wildcards)
        possible_answer[gt_trace['function_name']][param_name] = [param_value]

    return func_description, model_output, possible_answer


def check_ast_validity(tool_call: Dict[str, Any], ground_truth_trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a tool call is syntactically valid using BFCL's AST checker.

    Args:
        tool_call: Model's tool call {'name': 'func_name', 'arguments': {...}}
        ground_truth_trace: Ground truth execution trace

    Returns:
        Dict with 'valid' (bool), 'score' (float 0-1), 'error' (list)
    """
    func_desc, model_out, possible_ans = convert_to_bfcl_format(tool_call, ground_truth_trace)

    if func_desc is None:
        return {'valid': False, 'score': 0.0, 'error': ['No ground truth trace available']}

    # Use BFCL's simple_function_checker
    result = simple_function_checker(
        func_description=func_desc,
        model_output=model_out,
        possible_answer=possible_ans,
        language=Language.PYTHON,
        model_name='generic'  # Use generic model name
    )

    return {
        'valid': result['valid'],
        'score': 1.0 if result['valid'] else 0.0,
        'error': result.get('error', []),
        'error_type': result.get('error_type', '')
    }


def check_ast_validity_graduated(tool_call: Dict[str, Any], ground_truth_trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Graduated AST validation for RL training with partial credit.

    Returns scores for three levels:
    - Level 1 (0.33): Function name matches ground truth
    - Level 2 (0.33): Correct parameter structure and types
    - Level 3 (0.34): Correct parameter values (strict matching)

    Args:
        tool_call: Model's tool call {'name': 'func_name', 'arguments': {...}}
        ground_truth_trace: Ground truth execution trace

    Returns:
        Dict with scores for each level and total score
    """
    result = {
        'name_score': 0.0,
        'structure_score': 0.0,
        'value_score': 0.0,
        'total_score': 0.0,
        'details': {}
    }

    # Validate tool_call is a dict
    if not isinstance(tool_call, dict):
        result['details']['error'] = f'Tool call is {type(tool_call).__name__}, expected dict'
        return result

    # Check if ground truth trace is available
    if 'traces' not in ground_truth_trace or not ground_truth_trace['traces']:
        result['details']['error'] = 'No ground truth trace available'
        return result

    gt_trace = ground_truth_trace['traces'][0]
    tool_name = tool_call.get('name', '')

    # Level 1: Function name (0.33)
    if tool_name == gt_trace['function_name']:
        result['name_score'] = 0.33
        result['details']['name_match'] = True
    else:
        result['details']['name_match'] = False
        result['details']['expected_name'] = gt_trace['function_name']
        result['details']['actual_name'] = tool_name
        result['total_score'] = 0.0
        return result  # Early exit if name doesn't match

    # Level 2: Parameter structure and types (0.33)
    # Use safe_get_arguments to handle malformed tool calls
    model_args = safe_get_arguments(tool_call)
    gt_args = gt_trace['arguments']

    # Calculate parameter set overlap
    model_param_set = set(model_args.keys())
    gt_param_set = set(gt_args.keys())
    correct_params = model_param_set & gt_param_set
    missing_params = gt_param_set - model_param_set
    extra_params = model_param_set - gt_param_set

    result['details']['missing_params'] = list(missing_params)
    result['details']['extra_params'] = list(extra_params)

    # Edge case: no parameters required
    if len(gt_args) == 0:
        result['structure_score'] = 0.33 if len(model_args) == 0 else 0.0
        result['details']['structure_match'] = len(model_args) == 0
    else:
        # Calculate parameter overlap ratio
        param_overlap_ratio = len(correct_params) / len(gt_param_set)

        # Check types on overlapping parameters only
        type_matches = 0
        type_mismatches = []

        for param_name in correct_params:
            model_value = model_args[param_name]
            gt_value = gt_args[param_name]

            if types_compatible(model_value, gt_value):
                type_matches += 1
            else:
                type_mismatches.append({
                    'param': param_name,
                    'expected_type': type(gt_value).__name__,
                    'actual_type': type(model_value).__name__
                })

        # Calculate type accuracy on overlapping params
        if len(correct_params) > 0:
            type_accuracy = type_matches / len(correct_params)
        else:
            type_accuracy = 0.0

        # Combined score: overlap * type accuracy
        # Example: 4/5 params present with 3/4 correct types = 0.8 * 0.75 = 0.6 → 0.33 * 0.6 = 0.198
        result['structure_score'] = 0.33 * param_overlap_ratio * type_accuracy

        # Set match status
        if param_overlap_ratio == 1.0 and type_accuracy == 1.0:
            result['details']['structure_match'] = True
        elif param_overlap_ratio > 0 and type_accuracy > 0:
            result['details']['structure_match'] = 'partial'
            result['details']['param_overlap_ratio'] = param_overlap_ratio
            result['details']['type_accuracy'] = type_accuracy
        else:
            result['details']['structure_match'] = False

        if type_mismatches:
            result['details']['type_mismatches'] = type_mismatches

    # Level 3: Parameter values (0.34)
    # Use strict BFCL checker for value validation
    strict_result = check_ast_validity(tool_call, ground_truth_trace)

    if strict_result['valid']:
        result['value_score'] = 0.34
        result['details']['value_match'] = True
    else:
        result['value_score'] = 0.0
        result['details']['value_match'] = False
        result['details']['value_errors'] = strict_result.get('error', [])

    # Calculate total score
    result['total_score'] = result['name_score'] + result['structure_score'] + result['value_score']

    return result


def check_semantic_validity(
    tool_call: Dict[str, Any],
    tool_response: Dict[str, Any],
    ground_truth_trace: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if a tool call executed successfully based on actual execution response.

    Args:
        tool_call: Model's tool call {'name': 'func_name', 'arguments': {...}}
        tool_response: Actual execution result from the environment
        ground_truth_trace: Ground truth execution trace with expected results

    Returns:
        Dict with 'valid' (bool), 'score' (float 0-1), 'details' (str)
    """
    result = {
        'valid': False,
        'score': 0.0,
        'details': '',
        'checks': {}
    }

    # Check 1: Ground truth indicates this call should succeed
    if not ground_truth_trace.get('success', False):
        result['details'] = 'Ground truth indicates failure expected'
        result['checks']['gt_expects_success'] = False
        return result

    result['checks']['gt_expects_success'] = True

    # Check 2: Tool response exists
    if tool_response is None:
        result['details'] = 'No tool response found (tool may not have been executed)'
        result['checks']['has_response'] = False
        return result

    result['checks']['has_response'] = True

    # Check 3: Response doesn't contain error
    # Check for common error indicators in response
    if isinstance(tool_response, dict):
        if 'error' in tool_response:
            result['details'] = f"Response contains error: {tool_response['error']}"
            result['checks']['no_error'] = False
            return result

        # Also check for error in nested structures
        if 'status' in tool_response and tool_response['status'] == False:
            result['details'] = f"Response indicates failure: {tool_response.get('message', 'Unknown')}"
            result['checks']['no_error'] = False
            return result

    result['checks']['no_error'] = True

    # Check 4: Response has expected structure
    # For successful tool calls, we expect some data in the response
    # This is a lightweight check - we're not validating the actual content
    if isinstance(tool_response, dict):
        # Look for common success indicators
        has_data = any(key in tool_response for key in [
            'data', 'result', 'results', 'success', 'output',
            'hotels', 'flights', 'cars', 'attractions'  # BookingAPI specific
        ])

        if not has_data and len(tool_response) == 0:
            result['details'] = 'Response is empty dict'
            result['checks']['valid_structure'] = False
            return result

        result['checks']['valid_structure'] = True
    elif isinstance(tool_response, (list, str, int, float, bool)):
        # Primitive types are acceptable responses
        result['checks']['valid_structure'] = True
    else:
        result['details'] = f'Unexpected response type: {type(tool_response).__name__}'
        result['checks']['valid_structure'] = False
        return result

    # All checks passed - execution was successful
    result['valid'] = True
    result['score'] = 1.0
    result['details'] = 'Tool execution succeeded'

    return result


def compute_atomic_validity_reward(
    model_tool_calls: List[Dict[str, Any]],
    call_response_pairs: List[Tuple[Dict, Dict]],
    execution_traces: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute R_atomic_validity: rewards for individual function calls being correct.

    Uses graduated AST validation (name/structure/values) and actual execution results
    for semantic validation.

    Args:
        model_tool_calls: List of tool calls extracted from model output
        call_response_pairs: List of (call, response) tuples from actual execution
        execution_traces: Ground truth execution traces per step

    Returns:
        Dict with atomic validity scores and details
    """
    results = {
        'ast_scores': [],
        'semantic_scores': [],
        'name_scores': [],
        'structure_scores': [],
        'value_scores': [],
        'per_call_details': [],
        'total_calls': len(model_tool_calls),
        'valid_calls': 0,
        'executable_calls': 0,  # Calls that executed (had response)
        'successful_calls': 0   # Calls that executed without error
    }

    # Create response lookup that handles duplicate calls
    # Map from call signature to list of (index, response) tuples
    response_lookup = {}
    for idx, (call, response) in enumerate(call_response_pairs):
        if call and isinstance(call, dict):
            # Create signature from call content
            # Use safe_get_arguments to handle malformed tool calls
            arguments = safe_get_arguments(call)
            call_signature = (call.get('name', ''), str(sorted(arguments.items())))
            if call_signature not in response_lookup:
                response_lookup[call_signature] = []
            response_lookup[call_signature].append((idx, response))

    # Track which responses have been used (to handle duplicates correctly)
    used_response_indices = set()

    # Try to match each model tool call with ground truth traces
    for call_idx, tool_call in enumerate(model_tool_calls):
        # Handle malformed tool calls that aren't dicts
        if not isinstance(tool_call, dict):
            logger.warning(f"Tool call at index {call_idx} is not a dict: {type(tool_call).__name__} = {tool_call}")
            call_result = {
                'call_index': call_idx,
                'tool_name': 'malformed',
                'ast_valid': False,
                'semantic_valid': False,
                'ast_score': 0.0,
                'semantic_score': 0.0,
                'name_score': 0.0,
                'structure_score': 0.0,
                'value_score': 0.0,
                'matched_step': None,
                'error': f'Tool call is {type(tool_call).__name__}, expected dict'
            }
            results['per_call_details'].append(call_result)
            continue

        call_result = {
            'call_index': call_idx,
            'tool_name': tool_call.get('name', 'unknown'),
            'ast_valid': False,
            'semantic_valid': False,
            'ast_score': 0.0,
            'semantic_score': 0.0,
            'name_score': 0.0,
            'structure_score': 0.0,
            'value_score': 0.0,
            'matched_step': None
        }

        # Extract tool name and arguments for matching
        tool_name = tool_call.get('name', '')
        # Use safe_get_arguments to handle malformed tool calls
        arguments = safe_get_arguments(tool_call)

        # Find matching ground truth trace across all steps
        matched = False
        for step_idx, step_traces in enumerate(execution_traces):
            if tool_name in step_traces:
                gt_trace = step_traces[tool_name]

                # Check AST validity with graduated scoring
                ast_result = check_ast_validity_graduated(tool_call, gt_trace)
                call_result['ast_valid'] = (ast_result['total_score'] >= 1.0)
                call_result['ast_score'] = ast_result['total_score']
                call_result['name_score'] = ast_result['name_score']
                call_result['structure_score'] = ast_result['structure_score']
                call_result['value_score'] = ast_result['value_score']
                call_result['ast_details'] = ast_result['details']

                # Find corresponding response, handling duplicates
                # Reuse arguments extracted earlier to avoid malformed data issues
                call_signature = (tool_name, str(sorted(arguments.items())))
                tool_response = None

                if call_signature in response_lookup:
                    # Find first unused response for this signature
                    for resp_idx, response in response_lookup[call_signature]:
                        if resp_idx not in used_response_indices:
                            tool_response = response
                            used_response_indices.add(resp_idx)
                            break

                # Track tool calling success metrics
                if tool_response is not None:
                    results['executable_calls'] += 1
                    # Check if response doesn't contain error
                    has_error = False
                    if isinstance(tool_response, dict):
                        if 'error' in tool_response:
                            has_error = True
                        elif 'status' in tool_response and tool_response['status'] == False:
                            has_error = True
                    if not has_error:
                        results['successful_calls'] += 1

                # Check semantic validity using actual execution result
                semantic_result = check_semantic_validity(tool_call, tool_response, gt_trace)
                call_result['semantic_valid'] = semantic_result['valid']
                call_result['semantic_score'] = semantic_result['score']
                call_result['semantic_details'] = semantic_result['details']
                call_result['semantic_checks'] = semantic_result['checks']

                call_result['matched_step'] = step_idx
                matched = True
                break

        if not matched:
            call_result['error'] = f"No ground truth trace found for tool '{tool_call.get('name')}'"
            # Add zeros for unmatched calls
            call_result['ast_score'] = 0.0
            call_result['semantic_score'] = 0.0
            call_result['name_score'] = 0.0
            call_result['structure_score'] = 0.0
            call_result['value_score'] = 0.0

        # A call is fully valid if both AST and semantic checks pass
        if call_result['ast_valid'] and call_result['semantic_valid']:
            results['valid_calls'] += 1

        results['ast_scores'].append(call_result['ast_score'])
        results['semantic_scores'].append(call_result['semantic_score'])
        results['name_scores'].append(call_result['name_score'])
        results['structure_scores'].append(call_result['structure_score'])
        results['value_scores'].append(call_result['value_score'])
        results['per_call_details'].append(call_result)

    # Compute aggregate scores
    if results['total_calls'] > 0:
        results['avg_ast_score'] = sum(results['ast_scores']) / results['total_calls']
        results['avg_semantic_score'] = sum(results['semantic_scores']) / results['total_calls']
        results['avg_combined_score'] = (results['avg_ast_score'] + results['avg_semantic_score']) / 2
        results['validity_rate'] = results['valid_calls'] / results['total_calls']
        # Graduated scoring breakdowns
        results['avg_name_score'] = sum(results['name_scores']) / results['total_calls']
        results['avg_structure_score'] = sum(results['structure_scores']) / results['total_calls']
        results['avg_value_score'] = sum(results['value_scores']) / results['total_calls']
        # Tool calling success metrics
        results['executable_rate'] = results['executable_calls'] / results['total_calls']
        results['tool_calling_success_rate'] = results['successful_calls'] / results['total_calls']
    else:
        results['avg_ast_score'] = 0.0
        results['avg_semantic_score'] = 0.0
        results['avg_combined_score'] = 0.0
        results['validity_rate'] = 0.0
        results['avg_name_score'] = 0.0
        results['avg_structure_score'] = 0.0
        results['avg_value_score'] = 0.0
        results['executable_rate'] = 0.0
        results['tool_calling_success_rate'] = 0.0

    return results


def compute_model_reward(solutions, ground_truth, is_matched, weight_mode='uniform', step_dependency=None):
    """
    Compute step-level and global success rewards for model solutions vs ground truth.
    Now with dependency-aware ordering!
    
    Args:
        solutions: List of function calls [call_1, call_2, ...]
        ground_truth: List of steps [[step_1_call_1, step_1_call_2, ...], [step_2_...], ...]
        is_matched: Function that takes (func_call, gt_func_call) and returns True if matched
        weight_mode: 'uniform' for equal weights, 'dependency' for dependency-based weights
        step_dependency: List indicating dependencies, e.g., [None, [0], [0,1]]
    
    Returns:
        dict: Same as before but now respecting execution order
    """
    step_rewards = []
    step_weights = []
    step_matched_positions = {}  # Track which solution position matched each step
    
    for step_idx, gt_step in enumerate(ground_truth):
        R_step_i = 0.0
        matched_position = None
        
        # Find if any solution call matches this step
        for gt_call in gt_step:
            for sol_idx, sol_call in enumerate(solutions):
                if is_matched(sol_call, gt_call):
                    # Check dependency constraints
                    if step_dependency and step_dependency[step_idx] is not None:
                        # This step has dependencies - verify ordering
                        dependencies_satisfied = True
                        for dep_step_idx in step_dependency[step_idx]:
                            # Check if dependent step was matched
                            if dep_step_idx not in step_matched_positions:
                                dependencies_satisfied = False
                                break
                            # Check if dependent step came BEFORE current match
                            if step_matched_positions[dep_step_idx] >= sol_idx:
                                dependencies_satisfied = False
                                break
                        
                        if not dependencies_satisfied:
                            continue  # This match violates dependencies, keep searching
                    
                    # Valid match found!
                    R_step_i = 1.0
                    matched_position = sol_idx
                    break
            
            if R_step_i == 1.0:
                break
        
        # Record the matched position for dependency checking
        if matched_position is not None:
            step_matched_positions[step_idx] = matched_position
                
        step_rewards.append(R_step_i)
        
        # Calculate step weight
        if weight_mode == 'uniform':
            weight = 1.0
        elif weight_mode == 'dependency':
            weight = len(gt_step)
        else:
            weight = 1.0
            
        step_weights.append(weight)
    
    # Calculate global success reward as weighted average
    if sum(step_weights) > 0:
        weighted_sum = sum(r * w for r, w in zip(step_rewards, step_weights))
        R_success = weighted_sum / sum(step_weights)
    else:
        R_success = 0.0
    
    return {
        'step_rewards': step_rewards,
        'step_weights': step_weights,
        'global_reward': R_success,
        'matched_positions': step_matched_positions  # Useful for debugging
    }

def compute_score(solution_str, ground_truth, method="strict", extra_info=None, reward_weights=None, **kwargs):
    """Reward score function for complex tool orchestration.

    Computes three reward components:
    1. R_atomic_validity: Individual function call correctness (graduated AST + semantic)
    2. R_outcome_orchestration: Multi-step orchestration with dependency ordering
    3. R_outcome_state: Final state validation [PLACEHOLDER - always 0.0]

    Args:
        solution_str: the solution text from the model
        ground_truth: the ground truth tool orchestration
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        extra_info: dict with 'step_dependency' and 'execution_traces'
        reward_weights: dict with weights for each reward component. Defaults to:
            {'R_atomic_validity': 0.5, 'R_outcome_orchestration': 0.5, 'R_outcome_state': 0.0}
            Set any weight to 0.0 to disable that component (skips computation for efficiency)

    Returns:
        Dict with individual scores and total score
    """
    scores = {}

    # Set default weights if not provided
    if reward_weights is None:
        reward_weights = {
            'R_atomic_validity': 0.5,
            'R_outcome_orchestration': 0.5,
            'R_outcome_state': 0.0
        }

    # Extract tool calls and responses (needed by multiple components)
    extracted_results = extract_tool_calls_and_responses(solution_str)
    solution_tool_calls = extracted_results['tool_calls']
    call_response_pairs = extracted_results['call_response_pairs']

    # ==================================================================
    # R_atomic_validity: Check individual function calls (Graduated AST + Semantic)
    # ==================================================================
    # Skip computation if weight is 0.0 (performance optimization)
    if reward_weights.get('R_atomic_validity', 0.0) > 0.0:
        execution_traces = extra_info.get('execution_traces', []) if extra_info else []

        # Validate execution_traces are present (critical for R_atomic_validity)
        if not execution_traces and ground_truth:
            raise ValueError(
                "execution_traces is empty but ground_truth exists. "
                "Check that preprocessing includes 'execution_traces' in extra_info."
            )

        if execution_traces:
            atomic_validity_result = compute_atomic_validity_reward(
                model_tool_calls=solution_tool_calls,
                call_response_pairs=call_response_pairs,
                execution_traces=execution_traces
            )
            scores['R_atomic_validity'] = atomic_validity_result['avg_combined_score']
            # Add individual components as scalar metrics (not nested dicts)
            scores['R_atomic_validity_ast'] = atomic_validity_result['avg_ast_score']
            scores['R_atomic_validity_semantic'] = atomic_validity_result['avg_semantic_score']
            scores['R_atomic_validity_rate'] = atomic_validity_result['validity_rate']
            # Graduated AST breakdown
            scores['R_atomic_validity_name'] = atomic_validity_result['avg_name_score']
            scores['R_atomic_validity_structure'] = atomic_validity_result['avg_structure_score']
            scores['R_atomic_validity_values'] = atomic_validity_result['avg_value_score']
            # Tool calling success metrics
            scores['tool_executable_rate'] = atomic_validity_result['executable_rate']
            scores['tool_calling_success_rate'] = atomic_validity_result['tool_calling_success_rate']
        else:
            # Fallback if no execution traces available
            scores['R_atomic_validity'] = 0.0
            scores['R_atomic_validity_ast'] = 0.0
            scores['R_atomic_validity_semantic'] = 0.0
            scores['R_atomic_validity_rate'] = 0.0
            scores['R_atomic_validity_name'] = 0.0
            scores['R_atomic_validity_structure'] = 0.0
            scores['R_atomic_validity_values'] = 0.0
            scores['tool_executable_rate'] = 0.0
            scores['tool_calling_success_rate'] = 0.0
    else:
        # R_atomic_validity disabled (weight = 0.0)
        scores['R_atomic_validity'] = 0.0
        scores['R_atomic_validity_ast'] = 0.0
        scores['R_atomic_validity_semantic'] = 0.0
        scores['R_atomic_validity_rate'] = 0.0
        scores['R_atomic_validity_name'] = 0.0
        scores['R_atomic_validity_structure'] = 0.0
        scores['R_atomic_validity_values'] = 0.0
        scores['tool_executable_rate'] = 0.0
        scores['tool_calling_success_rate'] = 0.0

    # ==================================================================
    # R_outcome_orchestration: Multi-step orchestration with dependencies
    # ==================================================================
    # Skip computation if weight is 0.0 (performance optimization)
    if reward_weights.get('R_outcome_orchestration', 0.0) > 0.0:
        def is_matched(func_call, gt_func_call):
            assert 'name' in gt_func_call
            # Only match if both name matches AND this is the correct tool (success=True)
            if "name" in func_call and (func_call['name'] == gt_func_call['name']):
                if gt_func_call.get('success', False):
                    return True
            return False

        step_dependency = extra_info.get('step_dependency') if extra_info else None
        orchestration_reward = compute_model_reward(
            solution_tool_calls,
            ground_truth,
            is_matched,
            weight_mode="uniform",
            step_dependency=step_dependency
        )

        scores['R_outcome_orchestration'] = orchestration_reward['global_reward']
        # Note: step_rewards and matched_positions are not saved as they are complex structures
        # Only the global reward (scalar) is kept for metrics tracking
    else:
        # R_outcome_orchestration disabled (weight = 0.0)
        scores['R_outcome_orchestration'] = 0.0

    # ==================================================================
    # R_outcome_state: Final execution state validation [PLACEHOLDER]
    # ==================================================================
    # Skip computation if weight is 0.0 (performance optimization)
    if reward_weights.get('R_outcome_state', 0.0) > 0.0:
        # TODO: Implement state-based validation
        # - For tasks requiring file creation/modification: check file system state
        # - For information gathering: compare final response with expected answer
        # - For booking/reservation tasks: verify backend system state
        scores['R_outcome_state'] = 0.0  # Placeholder - not yet implemented
    else:
        scores['R_outcome_state'] = 0.0

    # ==================================================================
    # Total score: weighted combination of all components
    # ==================================================================
    # Use configurable weights passed via reward_weights parameter
    scores['score'] = (
        reward_weights.get('R_atomic_validity', 0.0) * scores['R_atomic_validity'] +
        reward_weights.get('R_outcome_orchestration', 0.0) * scores['R_outcome_orchestration'] +
        reward_weights.get('R_outcome_state', 0.0) * scores['R_outcome_state']
    )

    return scores
