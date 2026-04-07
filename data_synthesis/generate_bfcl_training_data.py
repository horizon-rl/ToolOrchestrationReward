#!/usr/bin/env python3
"""
Generate BFCL-based training data for RL using cache-constrained generation.

This script generates perfectly aligned training data by sampling complete parameter
sets from the BookingAPI cache. This ensures:
- 100% validation success (all cache hits)
- Perfect query-argument alignment
- Deterministic, reproducible training data

Workflow:
1. Load workflow templates from ComplexFuncBench
2. Load BookingAPI cache (complete function call entries)
3. For each workflow, sample ONE complete cache entry per independent step
4. LLM generates natural language query matching the exact parameters
5. Validate via BFCL execution (guaranteed cache hit)
6. Output training data with dependency tracking

Usage:
    python generate_bfcl_training_data.py \
        --target-size 250 \
        --model gpt-4o \
        --output data/bfcl_training.pkl
"""

import argparse
import json
import random
import sys
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup paths (repo root so `utils` resolves when run as `python data_synthesis/...`)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import BFCL utilities (BFCL should be installed as package)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

# Import project utilities
from utils.utils import get_llm_response, parse_json
from loguru import logger


class BFCLTrainingDataGenerator:
    """Generator for BFCL-based RL training data with cache-constrained generation."""

    def __init__(
        self,
        workflow_templates_path: str,
        model: str = "gpt-4.1",
        candidate_tools_per_step: int = 9,
        sampling_strategy: str = "uniform",
        random_seed: Optional[int] = None,
        num_distractors: int = 0
    ):
        """
        Initialize the generator.

        Args:
            workflow_templates_path: Path to workflow templates JSON
            model: LLM model for query generation
            candidate_tools_per_step: DEPRECATED - ignored, kept for backward compatibility.
                Candidate tools are now all tools from the workflow.
            sampling_strategy: "uniform" or "frequency_weighted"
            random_seed: Random seed for reproducibility
            num_distractors: Number of same-domain distractors to add per query (default: 0)
        """
        self.model = model
        self.candidate_tools_per_step = candidate_tools_per_step  # Kept for backward compatibility
        self.sampling_strategy = sampling_strategy
        self.num_distractors = num_distractors

        if random_seed is not None:
            random.seed(random_seed)

        # Load workflow templates
        logger.info(f"Loading workflow templates from {workflow_templates_path}")
        with open(workflow_templates_path) as f:
            self.templates_data = json.load(f)

        self.workflows = self.templates_data['workflows']
        self.all_functions = self.templates_data['metadata']['all_functions']
        self.dependency_patterns = self.templates_data['dependency_patterns']

        logger.info(f"Loaded {len(self.workflows)} workflow templates")
        logger.info(f"Available functions: {len(self.all_functions)}")

        # Load actual cache (complete parameter sets)
        cache_path = os.path.join(_repo_root, "environment", "booking_api_cache.json")
        logger.info(f"Loading BookingAPI cache from {cache_path}")
        with open(cache_path) as f:
            self.cache = json.load(f)

        total_entries = sum(len(entries) for entries in self.cache.values())
        logger.info(f"Loaded cache with {len(self.cache)} functions, {total_entries} complete entries")

        # Build reverse index for lazy compatibility checking
        logger.info("Building reverse index for cache-aware sampling...")
        self.reverse_index = self._build_reverse_index()
        logger.info(f"Reverse index built for lazy compatibility checking")

        # Get all available BookingAPI tools from cache for distractor sampling
        self.all_booking_tools = list(self.cache.keys())
        logger.info(f"Loaded {len(self.all_booking_tools)} BookingAPI tools from cache for distractor sampling")

    def _build_reverse_index(self) -> Dict[str, Dict[str, Dict[Any, List[int]]]]:
        """
        Build reverse index for lazy compatibility checking.

        For each function, index cache entries by parameter values.

        Returns:
            {
                'function_name': {
                    'param_name': {
                        'param_value': [cache_entry_index, ...],
                        ...
                    },
                    ...
                },
                ...
            }
        """
        reverse_index = {}

        for func_name, entries in self.cache.items():
            func_index = {}

            for idx, entry in enumerate(entries):
                args = entry['arguments']

                for param_name, param_value in args.items():
                    if param_name not in func_index:
                        func_index[param_name] = {}

                    # Convert value to hashable type
                    if isinstance(param_value, (list, dict)):
                        # Skip complex types for now
                        continue

                    if param_value not in func_index[param_name]:
                        func_index[param_name][param_value] = []

                    func_index[param_name][param_value].append(idx)

            reverse_index[func_name] = func_index

        return reverse_index

    def _extract_value_from_response(self, response: Any, field_path: str) -> Any:
        """
        Extract a value from a response using a field path.

        Handles the 'data' wrapper automatically if present in response.

        Examples:
            destinations[0].id  -> response.data.destinations[0].id
            products[0].name    -> response.data.products[0].name
            data.city           -> response.data.city
        """
        try:
            # Parse the field path
            import re
            tokens = re.findall(r'(\w+)|\[(\d+)\]', field_path)

            current = response

            # Auto-detect and navigate 'data' wrapper if present
            if isinstance(current, dict) and 'data' in current:
                # Check if field path starts with 'data'
                if tokens and tokens[0][0] != 'data':
                    # Field path doesn't start with 'data', but response has it
                    # Navigate into data automatically
                    current = current['data']

            for key, index in tokens:
                if index:
                    # Array index
                    current = current[int(index)]
                elif key:
                    # Object key
                    if isinstance(current, dict):
                        if key not in current:
                            return None
                        current = current[key]
                    else:
                        return None

            return current
        except (KeyError, IndexError, TypeError, AttributeError):
            return None

    def _sample_compatible_chain_lazy(self, workflow: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Lazily sample a compatible chain by building it step-by-step.

        For each step with dependencies, use the reverse index to find compatible entries.
        Retry up to max_retries times if no compatible entry is found.
        """
        max_retries = 50  # Increased from 10 to give more chances for sparse workflows

        for attempt in range(max_retries):
            sampled_entries = {}
            step_constraints = []
            failed = False

            for step_idx, dep_info in workflow['dependencies'].items():
                func_name = dep_info['function']
                step_idx_int = int(step_idx)

                if func_name not in self.cache:
                    logger.warning(f"Function {func_name} not in cache")
                    failed = True
                    break

                cache_entries = self.cache[func_name]
                if not cache_entries:
                    failed = True
                    break

                if not dep_info['depends_on']:
                    # Independent step - sample random entry
                    entry = random.choice(cache_entries)
                    sampled_entries[step_idx_int] = entry

                    step_constraints.append({
                        'step': step_idx_int,
                        'function': func_name,
                        'complete_args': entry['arguments'].copy(),
                        'is_dependent': False
                    })
                else:
                    # Dependent step - use reverse index to find compatible entry
                    dependency_args = dep_info['dependency_args']
                    compatible_indices = None

                    # Find entries that match ALL dependency requirements
                    for arg_name, dep_detail in dependency_args.items():
                        from_step = dep_detail['from_step']
                        from_field = dep_detail['from_field']

                        # Extract value from previous step
                        prev_entry = sampled_entries.get(from_step)
                        if not prev_entry:
                            failed = True
                            break

                        expected_value = self._extract_value_from_response(
                            prev_entry['response'],
                            from_field
                        )

                        if expected_value is None:
                            failed = True
                            break

                        # Look up compatible entries in reverse index
                        func_index = self.reverse_index.get(func_name, {})
                        param_index = func_index.get(arg_name, {})
                        matching_indices = param_index.get(expected_value, [])

                        if not matching_indices:
                            failed = True
                            break

                        # Intersect with previous constraints
                        if compatible_indices is None:
                            compatible_indices = set(matching_indices)
                        else:
                            compatible_indices &= set(matching_indices)

                    if failed or not compatible_indices:
                        failed = True
                        break

                    # Sample from compatible entries
                    entry_idx = random.choice(list(compatible_indices))
                    entry = cache_entries[entry_idx]
                    sampled_entries[step_idx_int] = entry

                    step_constraints.append({
                        'step': step_idx_int,
                        'function': func_name,
                        'complete_args': entry['arguments'].copy(),
                        'is_dependent': True,
                        'depends_on': dep_info['depends_on'],
                        'dependency_args': dep_info['dependency_args']
                    })

            if not failed:
                # Successfully built a compatible chain
                return step_constraints

        # Failed to find compatible chain after max retries
        logger.warning(f"Failed to find compatible chain for {workflow['id']} after {max_retries} attempts")
        return None

    def _build_constraints_from_chain(self, chain: Dict[str, Any], workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build step constraints from a compatible chain.

        Args:
            chain: Compatible chain with step{i}_entry keys
            workflow: Workflow definition

        Returns:
            List of step constraints
        """
        step_constraints = []

        for step_idx, dep_info in workflow['dependencies'].items():
            entry_key = f'step{step_idx}_entry'
            if entry_key not in chain:
                logger.error(f"Chain missing {entry_key}")
                return None

            entry = chain[entry_key]
            func_name = dep_info['function']
            complete_args = entry['arguments'].copy()

            # Determine if this step has dependencies
            if not dep_info['depends_on']:
                # Fully independent step - use all cache arguments
                step_constraints.append({
                    'step': int(step_idx),
                    'function': func_name,
                    'complete_args': complete_args,
                    'is_dependent': False
                })
            else:
                # Partially dependent step - use ALL cache params
                step_constraints.append({
                    'step': int(step_idx),
                    'function': func_name,
                    'complete_args': complete_args,
                    'is_dependent': True,
                    'depends_on': dep_info['depends_on'],
                    'dependency_args': dep_info['dependency_args']
                })

        return step_constraints

    def _sample_random_constraints(self, workflow: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Fallback: Sample random cache entries for each step.

        Used when no compatible chains are available.
        """
        step_constraints = []

        for step_idx, dep_info in workflow['dependencies'].items():
            func_name = dep_info['function']

            # Check if function exists in cache
            if func_name not in self.cache:
                logger.warning(f"Function {func_name} not found in cache - skipping workflow")
                return None

            # Sample ONE complete cache entry for this function
            cache_entries = self.cache[func_name]
            if not cache_entries:
                logger.warning(f"No cache entries for {func_name} - skipping workflow")
                return None

            sampled_entry = random.choice(cache_entries)
            complete_args = sampled_entry['arguments'].copy()

            # Determine if this step has dependencies
            if not dep_info['depends_on']:
                # Fully independent step - use all cache arguments
                step_constraints.append({
                    'step': int(step_idx),
                    'function': func_name,
                    'complete_args': complete_args,
                    'is_dependent': False
                })
            else:
                # Partially dependent step - use ALL cache params
                step_constraints.append({
                    'step': int(step_idx),
                    'function': func_name,
                    'complete_args': complete_args,
                    'is_dependent': True,
                    'depends_on': dep_info['depends_on'],
                    'dependency_args': dep_info['dependency_args']
                })

        return step_constraints

    def sample_workflow(self) -> Dict[str, Any]:
        """Sample a workflow according to the sampling strategy."""
        if self.sampling_strategy == "uniform":
            return random.choice(self.workflows)
        elif self.sampling_strategy == "frequency_weighted":
            weights = [w['frequency'] for w in self.workflows]
            return random.choices(self.workflows, weights=weights)[0]
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def generate_query_for_workflow(self, workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a synthetic query for a given workflow using cache-constrained generation.

        The LLM is given cached parameter values and must choose from them, ensuring
        perfect query-argument alignment and 100% validation success.

        In constrained mode, also generates the exact parameter values to use.
        In expand mode, generates freely and validates against cache (expanding if needed).

        Returns:
            {
                'query': natural language query,
                'workflow_id': workflow identifier,
                'ground_truth_steps': [
                    {
                        'step': step_index,
                        'function': function_name,
                        'arguments': {arg: value or '<FROM_STEP_X>'},
                        'depends_on': [step_indices],
                        'dependency_args': {arg_name: dependency_info}
                    }
                ]
            }
        """
        return self._generate_query(workflow)

    def _generate_query(self, workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate query using lazy cache-compatible sampling.

        Uses reverse index to lazily find compatible cache entries step-by-step.
        """
        step_constraints = self._sample_compatible_chain_lazy(workflow)

        if step_constraints is None:
            return None

        # Build prompt with exact arguments (no choices)
        system_msg = (
            "You are generating natural language queries for a travel booking assistant.\n\n"
            "You will be given EXACT API parameter values to use (from validated cache).\n"
            "Your task is ONLY to generate a natural language query that matches these exact parameters.\n"
            "DO NOT modify the parameter values - use them exactly as provided."
        )

        # Format step parameters for LLM
        parameters_text = ""
        for step in step_constraints:
            parameters_text += f"\nStep {step['step']} - {step['function']}:\n"
            if step['is_dependent']:
                parameters_text += f"  (Parameters will be extracted from previous step results)\n"
            else:
                for param, value in step['complete_args'].items():
                    # Format value for display
                    if isinstance(value, str):
                        parameters_text += f"  {param}: '{value}'\n"
                    else:
                        parameters_text += f"  {param}: {value}\n"

        user_msg = f"""
Workflow pattern: {' -> '.join(workflow['pattern'])}

EXACT PARAMETERS TO USE (do not modify):
{parameters_text}

Task:
Generate a natural language query that matches these exact parameter values.

OUTPUT FORMAT (JSON):
{{
    "query": "Natural language query that matches the given parameters",
    "chosen_parameters": [
        {{
            "step": 0,
            "function": "{workflow['pattern'][0]}",
            "arguments": {{"param1": value1, ...}}  // Echo back the exact parameters provided
        }},
        ...
    ],
    "variation_notes": "Brief description of the scenario"
}}

IMPORTANT: The query must match the exact parameters provided above!
"""

        try:
            prompt = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]

            response = get_llm_response(prompt, model=self.model, return_type="text_only")
            parsed = parse_json(response)

            if 'query' not in parsed:
                logger.warning(f"LLM response missing 'query' field")
                return None

            # Build ground truth steps using sampled cache arguments
            # IMPORTANT: complete_args already contains ALL parameters (including dependent ones)
            # from the sampled cache chain, so we use them directly for guaranteed cache hits
            ground_truth_steps = []
            for step_constraint in step_constraints:
                step_idx = step_constraint['step']
                dep_info = workflow['dependencies'][str(step_idx)]

                # Use complete_args directly - they are already sampled from cache
                # This ensures 100% cache hit rate during validation
                arguments = step_constraint['complete_args'].copy()

                ground_truth_steps.append({
                    'step': step_idx,
                    'function': step_constraint['function'],
                    'arguments': arguments,
                    'depends_on': dep_info['depends_on'],
                    'dependency_args': dep_info.get('dependency_args', {})
                })

            return {
                'query': parsed['query'],
                'workflow_id': workflow['id'],
                'workflow_pattern': workflow['pattern'],
                'ground_truth_steps': ground_truth_steps,
                'variation_notes': parsed.get('variation_notes', '')
            }

        except Exception as e:
            logger.error(f"Error generating constrained query for workflow {workflow['id']}: {e}")
            return None

    def _generate_constrained_query(self, workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate query in constrained mode using only cached parameter values."""
        # Extract available parameter values for each step
        step_constraints = []

        for step_idx, dep_info in workflow['dependencies'].items():
            func_name = dep_info['function']

            if func_name not in self.cache_values:
                logger.warning(f"Function {func_name} not found in cache - skipping workflow")
                return None

            # Get available values for each parameter
            param_options = {}
            for param_name, param_value in dep_info['arguments'].items():
                # Skip dependent parameters (will be filled from previous steps)
                if param_name in dep_info['dependency_args']:
                    param_options[param_name] = "<depends_on_previous_step>"
                    continue

                # Get cached values for this parameter
                if param_name in self.cache_values[func_name]:
                    param_info = self.cache_values[func_name][param_name]
                    # Sample a few values to give LLM options
                    available_values = param_info['values']
                    sample_size = min(10, len(available_values))
                    sampled_values = random.sample(available_values, sample_size)
                    param_options[param_name] = sampled_values
                else:
                    logger.warning(f"Parameter {param_name} not found in cache for {func_name}")
                    return None

            step_constraints.append({
                'step': int(step_idx),
                'function': func_name,
                'param_options': param_options
            })

        # Build constrained prompt
        system_msg = (
            "You are generating natural language queries AND matching API parameters for a travel booking assistant.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- You MUST use ONLY the parameter values provided in the constraints\n"
            "- For each step, you are given a list of valid parameter values to choose from\n"
            "- Pick ONE value from each provided list\n"
            "- The natural language query must match the chosen parameter values exactly\n"
            "- For dependent parameters (marked as '<depends_on_previous_step>'), do NOT provide values - they will be filled from previous step results"
        )

        # Format step constraints for LLM
        constraints_text = ""
        for step in step_constraints:
            constraints_text += f"\nStep {step['step']} - {step['function']}:\n"
            for param, options in step['param_options'].items():
                if options == "<depends_on_previous_step>":
                    constraints_text += f"  {param}: (filled from previous step)\n"
                else:
                    # Show sample of options
                    sample_display = options[:5]
                    more = f" ... ({len(options)} total options)" if len(options) > 5 else ""
                    constraints_text += f"  {param}: Choose ONE from: {sample_display}{more}\n"

        user_msg = f"""
Workflow pattern: {' -> '.join(workflow['pattern'])}

Parameter constraints (you MUST use values from these lists):
{constraints_text}

Task:
1. Choose specific parameter values from the provided lists for each step
2. Generate a natural language query that matches these exact parameter values
3. Return both the query and the chosen parameter values

OUTPUT FORMAT (JSON):
{{
    "query": "Natural language query that matches the chosen parameters",
    "chosen_parameters": [
        {{
            "step": 0,
            "function": "{workflow['pattern'][0]}",
            "arguments": {{"param1": "chosen_value1", ...}}  // Only non-dependent params
        }},
        ...
    ],
    "variation_notes": "Brief description of what scenario you created"
}}

IMPORTANT: The query and parameters must be perfectly aligned!
"""

        try:
            prompt = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]

            response = get_llm_response(prompt, model=self.model, return_type="text_only")
            parsed = parse_json(response)

            if 'query' not in parsed or 'chosen_parameters' not in parsed:
                logger.warning(f"LLM response missing required fields")
                return None

            # Build ground truth steps using chosen parameters
            ground_truth_steps = []
            for step_idx, dep_info in workflow['dependencies'].items():
                step_idx = int(step_idx)

                # Find chosen parameters for this step
                chosen_step = next(
                    (s for s in parsed['chosen_parameters'] if s['step'] == step_idx),
                    None
                )
                if chosen_step is None:
                    logger.warning(f"Missing chosen parameters for step {step_idx}")
                    return None

                # Merge chosen params with dependent args
                arguments = chosen_step['arguments'].copy()

                # Add dependent arguments (symbolic + value from template for validation)
                for arg_name in dep_info['dependency_args'].keys():
                    dep_detail = dep_info['dependency_args'][arg_name]
                    from_step = dep_detail['from_step']
                    arguments[arg_name] = {
                        'symbolic': f'<FROM_STEP_{from_step}>',
                        'value': dep_detail['value'],  # Will be replaced during validation
                        'from_field': dep_detail['from_field']
                    }

                ground_truth_steps.append({
                    'step': step_idx,
                    'function': dep_info['function'],
                    'arguments': arguments,
                    'depends_on': dep_info['depends_on'],
                    'dependency_args': dep_info['dependency_args']
                })

            return {
                'query': parsed['query'],
                'workflow_id': workflow['id'],
                'workflow_pattern': workflow['pattern'],
                'ground_truth_steps': ground_truth_steps,
                'variation_notes': parsed.get('variation_notes', '')
            }

        except Exception as e:
            logger.error(f"Error generating constrained query for workflow {workflow['id']}: {e}")
            return None

    def extract_field_value(self, step_result: Dict[str, Any], field_path: str) -> Any:
        """
        Extract value from nested API execution result using path notation.

        Args:
            step_result: Step execution result dict
            field_path: Path like "destinations[0].id" or "products[0].slug"

        Returns:
            Extracted value

        Raises:
            ValueError: If field path cannot be resolved
        """
        import re

        # Navigate to data section (handle double nesting from BFCL)
        if 'execution_result' in step_result:
            exec_result = step_result['execution_result']
            if 'execution_result' in exec_result:
                # Double nested: step_result['execution_result']['execution_result']['data']
                data = exec_result['execution_result'].get('data', {})
            else:
                # Single nested: step_result['execution_result']['data']
                data = exec_result.get('data', {})
        else:
            # Direct data access
            data = step_result.get('data', step_result)

        if not field_path or not data:
            return data

        # Parse path into tokens: keys and array indices
        # Matches: "destinations", "[0]", "id", "[123]"
        tokens = re.findall(r'(\w+)|\[(\d+)\]', field_path)
        current = data
        path_so_far = "data"

        try:
            for key, idx in tokens:
                if key:
                    # Dictionary key access
                    current = current[key]
                    path_so_far += f".{key}"
                elif idx:
                    # Array index access
                    index = int(idx)
                    current = current[index]
                    path_so_far += f"[{index}]"

            return current

        except (KeyError, IndexError, TypeError) as e:
            # Provide helpful error message for debugging
            if isinstance(current, dict):
                available = f"dict with keys: {list(current.keys())[:5]}"
            elif isinstance(current, list):
                available = f"list of length {len(current)}"
            else:
                available = f"{type(current).__name__}"

            raise ValueError(
                f"Failed to extract '{field_path}' at {path_so_far}\n"
                f"  Available: {available}\n"
                f"  Original error: {type(e).__name__}: {e}"
            )

    def validate_with_bfcl(self, query_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate a generated query by executing via BFCL.

        Returns validation results or None if validation fails.
        """
        logger.debug(f"Validating query: {query_data['query'][:100]}...")

        validation_results = []
        accumulated_results = {}  # Store results for dependent steps

        for step in query_data['ground_truth_steps']:
            step_idx = step['step']
            func_name = step['function']
            arguments = step['arguments'].copy()

            # Arguments are already complete from cache sampling - no need to resolve dependencies
            # The complete_args from _sample_compatible_chain_lazy already contains all values
            # including dependent parameters that were matched via reverse index
            #
            # Legacy code for resolving from response is no longer needed since we use
            # pre-sampled cache chains that guarantee all parameters are cache-compatible

            # Format for BFCL execution with type-aware conversion
            formatted_args = []
            for k, v in arguments.items():
                if isinstance(v, str):
                    # Strings - use repr to get quoted format
                    formatted_args.append(f'{k}={repr(v)}')
                elif isinstance(v, bool):
                    # Booleans - must check before int (bool is subclass of int)
                    formatted_args.append(f'{k}={v}')
                elif isinstance(v, (int, float)):
                    # Numbers - keep as-is (coordinates need to be numbers, not strings)
                    formatted_args.append(f'{k}={v}')
                elif isinstance(v, list):
                    # Lists - use Python list syntax (not JSON string)
                    formatted_args.append(f'{k}={repr(v)}')
                elif isinstance(v, dict):
                    # Dicts - use Python dict syntax
                    formatted_args.append(f'{k}={repr(v)}')
                elif v is None:
                    # None values
                    formatted_args.append(f'{k}=None')
                else:
                    # Fallback for other types
                    formatted_args.append(f'{k}={repr(v)}')

            arg_str = ', '.join(formatted_args)
            func_call_str = f'{func_name}({arg_str})'

            # DEBUG: Log the exact function call being executed
            logger.debug(f"Executing BFCL call: {func_call_str[:200]}")

            try:
                # Execute via BFCL (cache-only, guaranteed to succeed)
                # Use fixed test_entry_id to reuse the same BookingAPI instance
                # This avoids reloading the 24GB cache on every call
                results, _ = execute_multi_turn_func_call(
                    func_call_list=[func_call_str],
                    initial_config={"BookingAPI": {}},  # Cache-only mode
                    involved_classes=["BookingAPI"],
                    model_name="synthetic_generator",
                    test_entry_id="shared_instance",  # Fixed ID to reuse instance
                    long_context=False,
                    is_evaL_run=False
                )

                execution_result = results[0]

                # BFCL may return string representation - convert to dict/list
                if isinstance(execution_result, str):
                    try:
                        # Try to parse as JSON first
                        execution_result = json.loads(execution_result)
                    except json.JSONDecodeError:
                        # Try Python literal eval (for list/dict string representations)
                        try:
                            import ast
                            execution_result = ast.literal_eval(execution_result)
                        except (ValueError, SyntaxError):
                            logger.error(f"Failed to parse BFCL result: {execution_result[:200]}...")
                            return None

                # Check if execution succeeded
                # Cache format is uniform: {'status': bool, 'message': str, 'data': ...}
                # Success = status is True and data is non-empty
                if isinstance(execution_result, list):
                    # List response (some APIs return list directly)
                    success = len(execution_result) > 0
                elif isinstance(execution_result, dict):
                    # Standard cache format: check status and data
                    has_status_true = execution_result.get('status') is True
                    has_data = bool(execution_result.get('data'))
                    has_error = 'error' in execution_result
                    success = has_status_true and has_data and not has_error
                else:
                    success = False

                if not success:
                    msg = execution_result.get('message', 'Unknown error') if isinstance(execution_result, dict) else str(execution_result)
                    # Log full error for debugging
                    if isinstance(execution_result, dict):
                        error_detail = execution_result.get('data', {})
                        # Show both the function call string and the error
                        logger.warning(f"Step {step_idx} execution failed: {func_call_str[:150]}")
                        logger.warning(f"  Error: {msg}")
                        if error_detail:
                            logger.warning(f"  Detail: {error_detail}")
                        # Show full execution result structure for debugging
                        logger.warning(f"  Full result: {json.dumps(execution_result, indent=2, default=str)[:500]}")
                    else:
                        logger.warning(f"Step {step_idx} execution failed: {func_name}")
                        logger.warning(f"  Result: {str(execution_result)[:300]}")
                    return None  # Fail fast - query not valid

                validation_results.append({
                    'step': step_idx,
                    'function': func_name,
                    'arguments': arguments,
                    'execution_result': execution_result,
                    'success': success
                })

                accumulated_results[step_idx] = execution_result

            except Exception as e:
                logger.error(f"BFCL execution error at step {step_idx}: {e}")
                return None

        # All steps validated successfully
        return {
            'overall_success': True,
            'steps': validation_results,
            'query': query_data['query'],
            'workflow_id': query_data['workflow_id']
        }

    def select_candidate_tools(self, step_index: int, workflow_pattern: List[str]) -> List[str]:
        """
        Select candidate tools for a given step.

        Returns all unique tools from the workflow plus optional same-domain distractors.
        This creates confusors that test:
        1. Temporal reasoning (same-workflow tools at wrong time)
        2. Tool discrimination (similar but irrelevant tools)

        Args:
            step_index: Current step index (unused, kept for API compatibility)
            workflow_pattern: List of function names in the workflow

        Returns:
            List of candidate tool names (workflow + distractors)
        """
        # Start with all unique workflow tools
        workflow_tools = list(set(workflow_pattern))

        if self.num_distractors == 0:
            return workflow_tools

        # Sample same-domain distractors (tools NOT in workflow)
        available_distractors = [
            tool for tool in self.all_booking_tools
            if tool not in workflow_tools
        ]

        # Sample up to num_distractors (or fewer if not enough available)
        actual_num = min(self.num_distractors, len(available_distractors))
        if actual_num > 0:
            distractors = random.sample(available_distractors, actual_num)
            return workflow_tools + distractors
        else:
            return workflow_tools

    def format_for_rl_training(
        self,
        query_data: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format validated query into RL training data format.

        Output format matches preprocess_rl_data.py expectations.
        """
        steps = []

        for step_info in query_data['ground_truth_steps']:
            step_idx = step_info['step']

            # Select candidate tools for this step
            candidate_tools = self.select_candidate_tools(
                step_idx,
                query_data['workflow_pattern']
            )

            # Get actual arguments used during execution (after dependency resolution)
            executed_args = validation_results['steps'][step_idx]['arguments']

            # Prepare ground truth with both symbolic and actual values
            ground_truth_args = {}
            for arg_name, arg_value in step_info['arguments'].items():
                if isinstance(arg_value, dict) and 'from_field' in arg_value:
                    # Dependent argument - use actual value from execution
                    ground_truth_args[arg_name] = {
                        'symbolic_ref': arg_value.get('symbolic', f"<FROM_STEP_{step_info['depends_on'][0]}>"),
                        'actual_value': executed_args[arg_name],  # Actual value used during execution
                        'from_field': arg_value['from_field']
                    }
                else:
                    # Independent argument - use as-is
                    ground_truth_args[arg_name] = arg_value

            steps.append({
                'step': step_idx,
                'ground_truth': {
                    'function': step_info['function'],
                    'arguments': ground_truth_args
                },
                'candidate_tools': candidate_tools,
                'depends_on': step_info['depends_on'],
                'execution_result': validation_results['steps'][step_idx]
            })

        return {
            'query': query_data['query'],
            'workflow_id': query_data['workflow_id'],
            'workflow_pattern': query_data['workflow_pattern'],
            'steps': steps,
            'overall_success': True,
            'metadata': {
                'model': self.model,
                'candidate_selection': {
                    'mode': 'workflow_only' if self.num_distractors == 0 else 'same_domain_distractor',
                    'num_distractors': self.num_distractors
                },
                'sampling_strategy': self.sampling_strategy
            }
        }

    def generate_dataset(
        self,
        target_size: int,
        max_iterations: int = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset of validated training samples.

        Args:
            target_size: Target number of successful samples
            max_iterations: Maximum attempts (default: target_size * 3)

        Returns:
            List of training samples
        """
        if max_iterations is None:
            max_iterations = target_size * 3

        dataset = []
        attempts = 0
        workflow_distribution = defaultdict(int)

        logger.info(f"Generating dataset: target={target_size}, max_iterations={max_iterations}")

        while len(dataset) < target_size and attempts < max_iterations:
            attempts += 1

            # Sample workflow
            workflow = self.sample_workflow()

            # Generate query
            query_data = self.generate_query_for_workflow(workflow)
            if query_data is None:
                continue

            # Validate with BFCL
            validation_results = self.validate_with_bfcl(query_data)
            if validation_results is None:
                continue

            # Format for RL training
            training_sample = self.format_for_rl_training(query_data, validation_results)

            dataset.append(training_sample)
            workflow_distribution[workflow['id']] += 1

            if len(dataset) % 10 == 0:
                logger.info(f"Progress: {len(dataset)}/{target_size} samples generated ({attempts} attempts)")

        logger.success(f"Dataset generation complete: {len(dataset)} samples from {attempts} attempts")
        logger.info(f"Success rate: {len(dataset)/attempts*100:.1f}%")

        # Print workflow distribution
        logger.info("Workflow distribution:")
        for wf_id in sorted(workflow_distribution.keys(), key=lambda x: workflow_distribution[x], reverse=True)[:10]:
            count = workflow_distribution[wf_id]
            logger.info(f"  {wf_id}: {count} samples")

        return dataset

    def generate_dataset_per_workflow(
        self,
        queries_per_workflow: int,
        max_attempts_per_workflow: int = None,
        output_path: str = None,
        save_every_n_workflows: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset with fixed number of queries per workflow.

        Iterates through all workflows and generates queries_per_workflow samples
        for each workflow, ensuring balanced coverage.

        Args:
            queries_per_workflow: Number of queries to generate per workflow
            max_attempts_per_workflow: Max attempts per workflow (default: queries_per_workflow * 5)
            output_path: Path to save intermediate results (optional)
            save_every_n_workflows: Save checkpoint every N workflows (default: 10)

        Returns:
            List of training samples
        """
        if max_attempts_per_workflow is None:
            max_attempts_per_workflow = queries_per_workflow * 5

        dataset = []
        workflow_stats = {}
        total_workflows = len(self.workflows)

        logger.info(f"Generating dataset: {queries_per_workflow} queries per workflow, {total_workflows} workflows")
        logger.info(f"Expected total: ~{queries_per_workflow * total_workflows} samples")
        if output_path:
            logger.info(f"Checkpoints will be saved every {save_every_n_workflows} workflows to {output_path}")

        for wf_idx, workflow in enumerate(self.workflows):
            wf_id = workflow['id']
            wf_samples = []
            attempts = 0

            while len(wf_samples) < queries_per_workflow and attempts < max_attempts_per_workflow:
                attempts += 1

                # Generate query for this specific workflow
                query_data = self.generate_query_for_workflow(workflow)
                if query_data is None:
                    continue

                # Validate with BFCL
                validation_results = self.validate_with_bfcl(query_data)
                if validation_results is None:
                    continue

                # Format for RL training
                training_sample = self.format_for_rl_training(query_data, validation_results)
                wf_samples.append(training_sample)

            # Record stats
            workflow_stats[wf_id] = {
                'generated': len(wf_samples),
                'attempts': attempts,
                'success_rate': len(wf_samples) / attempts * 100 if attempts > 0 else 0
            }

            dataset.extend(wf_samples)

            # Progress update and checkpoint save
            if (wf_idx + 1) % 10 == 0 or wf_idx == total_workflows - 1:
                logger.info(f"Progress: {wf_idx + 1}/{total_workflows} workflows, {len(dataset)} total samples")

                # Save checkpoint
                if output_path and (wf_idx + 1) % save_every_n_workflows == 0:
                    checkpoint_path = Path(output_path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(dataset, f)
                    logger.info(f"Checkpoint saved: {len(dataset)} samples to {checkpoint_path}")

        # Summary
        successful_workflows = sum(1 for s in workflow_stats.values() if s['generated'] >= queries_per_workflow)
        partial_workflows = sum(1 for s in workflow_stats.values() if 0 < s['generated'] < queries_per_workflow)
        failed_workflows = sum(1 for s in workflow_stats.values() if s['generated'] == 0)

        logger.success(f"Dataset generation complete: {len(dataset)} samples from {total_workflows} workflows")
        logger.info(f"Workflow coverage: {successful_workflows} full, {partial_workflows} partial, {failed_workflows} failed")

        # Show workflows with issues
        if failed_workflows > 0 or partial_workflows > 0:
            logger.warning("Workflows with incomplete samples:")
            for wf_id, stats in sorted(workflow_stats.items(), key=lambda x: x[1]['generated']):
                if stats['generated'] < queries_per_workflow:
                    logger.warning(f"  {wf_id}: {stats['generated']}/{queries_per_workflow} ({stats['attempts']} attempts)")

        return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate BFCL-based training data for RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--workflow-templates",
        type=str,
        default="data/workflow_templates.json",
        help="Path to workflow templates JSON"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/bfcl_training.pkl",
        help="Output path for training data"
    )

    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Target number of training samples (random sampling mode)"
    )

    parser.add_argument(
        "--queries-per-workflow",
        type=int,
        default=None,
        help="Number of queries to generate per workflow (per-workflow mode). "
             "If set, ignores --target-size and generates queries for each of the 107 workflows."
    )

    parser.add_argument(
        "--model",
        type=str,
        # default="gpt-4.1",
        default="claude-v4-sonnet",
        help="LLM model for query generation"
    )

    parser.add_argument(
        "--candidate-tools",
        type=int,
        default=9,
        help="DEPRECATED: kept for backward compatibility, use --num-distractors instead"
    )

    parser.add_argument(
        "--num-distractors",
        type=int,
        default=0,
        help="Number of same-domain distractors to add per query (default: 0 for workflow-only)"
    )

    parser.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["uniform", "frequency_weighted"],
        default="uniform",
        help="Workflow sampling strategy"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    if args.verbose:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    else:
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Initialize generator
    generator = BFCLTrainingDataGenerator(
        workflow_templates_path=args.workflow_templates,
        model=args.model,
        candidate_tools_per_step=args.candidate_tools,
        sampling_strategy=args.sampling_strategy,
        random_seed=args.random_seed,
        num_distractors=args.num_distractors
    )

    # Generate dataset
    if args.queries_per_workflow is not None:
        # Per-workflow mode: generate fixed number of queries per workflow
        dataset = generator.generate_dataset_per_workflow(
            queries_per_workflow=args.queries_per_workflow,
            output_path=args.output,  # For checkpoint saving
            save_every_n_workflows=10
        )
    elif args.target_size is not None:
        # Random sampling mode: generate target number of samples
        dataset = generator.generate_dataset(target_size=args.target_size)
    else:
        # Default: 250 samples random sampling
        dataset = generator.generate_dataset(target_size=250)

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    logger.success(f"Training data saved to {output_path}")
    logger.info(f"Total samples: {len(dataset)}")

    return 0


if __name__ == "__main__":
    exit(main())
