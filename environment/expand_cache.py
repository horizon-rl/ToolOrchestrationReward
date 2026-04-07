#!/usr/bin/env python3
"""
Expand BookingAPI cache by generating diverse parameter combinations.

This script focuses purely on cache expansion - it does NOT generate training queries.
Instead, it uses LLM to generate diverse parameter values, executes the APIs directly,
and caches the results.

Purpose: Expand booking_api_cache.json beyond the 5,070 ComplexFuncBench entries.

Requirements:
- RAPID_API_KEY environment variable (for live API calls)
- Existing cache file (will be expanded, not overwritten)

Workflow:
1. Load existing cache to avoid duplicates
2. For each selected function, LLM generates diverse parameter combinations
3. Execute APIs directly via BookingAPI
4. Cache successful responses automatically
5. Report statistics (attempts, successes, cache growth)

Usage:
    # Expand cache for specific functions
    python expand_cache.py \
        --functions Search_Hotels,Search_Flights \
        --samples-per-function 50 \
        --model gpt-4o

    # Expand cache for all functions
    python expand_cache.py \
        --functions all \
        --samples-per-function 100
"""

import argparse
import inspect
import json
import os
import random
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Setup paths: repo root for local imports and repo-relative defaults
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import utilities
from utils.utils import get_llm_response, parse_json
from loguru import logger

# Use the repo-local BookingAPI implementation.
from environment.booking_api import BookingAPI


class CacheExpander:
    """Expand BookingAPI cache with diverse parameter combinations."""

    def __init__(
        self,
        model: str = "gpt-5.1",
        cache_path: Optional[str] = None,
        tool_schema_path: Optional[str] = None,
        workflow_templates_path: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the cache expander.

        Args:
            model: LLM model for parameter generation
            cache_path: Path to booking_api_cache.json (will be updated in-place)
            tool_schema_path: Path to booking_api.json (tool schemas)
            workflow_templates_path: Path to workflow_templates.json (for workflow mode)
            random_seed: Random seed for reproducibility
        """
        self.model = model

        if random_seed is not None:
            random.seed(random_seed)

        # Default paths
        if cache_path is None:
            cache_path = os.path.join(_repo_root, "environment", "booking_api_cache.json")
        if tool_schema_path is None:
            tool_schema_path = os.path.join(
                _repo_root,
                "environment/booking_api.json"
            )

        self.cache_path = cache_path
        self.tool_schema_path = tool_schema_path

        # Load tool schemas (JSONL format - one JSON object per line)
        logger.info(f"Loading tool schemas from {tool_schema_path}")
        self.tool_schemas = {}
        with open(tool_schema_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    func = json.loads(line)
                    self.tool_schemas[func['name']] = func
        logger.info(f"Loaded {len(self.tool_schemas)} function schemas")

        # Load existing cache
        logger.info(f"Loading existing cache from {cache_path}")
        with open(cache_path) as f:
            self.existing_cache = json.load(f)

        # Count existing entries per function
        self.cache_stats = defaultdict(int)
        for cache_key in self.existing_cache.keys():
            func_name = cache_key.split("(")[0]
            self.cache_stats[func_name] += 1

        logger.info(f"Existing cache: {len(self.existing_cache)} total entries")
        logger.info(f"Coverage: {len(self.cache_stats)} functions")

        # Initialize BookingAPI with live mode
        if "RAPID_API_KEY" not in os.environ:
            raise ValueError(
                "RAPID_API_KEY not found in environment. "
                "Set it with: export RAPID_API_KEY='your_key_here'"
            )

        logger.info("Initializing BookingAPI in LIVE mode (requires RAPID_API_KEY)")
        # Disable auto-save for performance (we'll save periodically instead)
        self.booking_api = BookingAPI(use_live_cache=True, auto_save_cache=False)
        logger.info("✓ BookingAPI initialized in LIVE mode (batch save enabled)")

        # Track when we last saved the cache
        self.last_cache_save_time = time.time()
        self.workflows_since_last_save = 0
        self.cache_save_interval = 3600  # Save every 1 hour
        self.cache_save_workflow_interval = 50  # Save every 50 workflows

        # Thread lock for cache access (for parallel execution)
        self.cache_lock = threading.Lock()

        # Shutdown flag for graceful interruption
        self._shutdown_requested = False

        # Track expansion statistics
        self.expansion_stats = {
            'attempts': defaultdict(int),
            'successes': defaultdict(int),
            'failures': defaultdict(int),
            'duplicates': defaultdict(int),
            'new_entries': defaultdict(int),
            'cache_hits': defaultdict(int),  # Track cache hits
            'api_timings': []  # Track API call durations for performance analysis
        }

        # Load historical workflow chain counts from summary files
        self.historical_chain_counts = self._load_historical_chain_counts()

        # Track initial cache size for hit rate calculation
        self.initial_cache_size = len(self.existing_cache)

        # Load workflow templates (for workflow mode)
        self.workflows = None
        self.templates_data = None
        if workflow_templates_path is not None:
            if workflow_templates_path == "default":
                workflow_templates_path = os.path.join(_repo_root, "data/workflow_templates.json")

            logger.info(f"Loading workflow templates from {workflow_templates_path}")
            try:
                with open(workflow_templates_path) as f:
                    self.templates_data = json.load(f)
                self.workflows = self.templates_data['workflows']
                logger.info(f"Loaded {len(self.workflows)} workflow templates")
            except FileNotFoundError:
                logger.warning(f"Workflow templates not found at {workflow_templates_path}")
                logger.warning("Workflow mode will not be available")
            except Exception as e:
                logger.error(f"Error loading workflow templates: {e}")
                logger.warning("Workflow mode will not be available")

    def _load_historical_chain_counts(self) -> Dict[str, int]:
        """
        Load historical chain counts from all summary files.

        Returns:
            Dict mapping workflow_id to total chains generated across all runs
        """
        import glob

        chain_counts = defaultdict(int)
        summary_dir = "logs/cache_expansion"

        if not os.path.exists(summary_dir):
            return chain_counts

        # Find all summary files
        summary_files = glob.glob(f"{summary_dir}/summary_*.json")

        for summary_file in summary_files:
            try:
                with open(summary_file) as f:
                    summary = json.load(f)

                # Extract chain counts from workflow_details
                workflow_details = summary.get('workflow_details', {})
                for workflow_id, details in workflow_details.items():
                    num_chains = details.get('num_chains', 0)
                    if num_chains > 0:
                        chain_counts[workflow_id] += num_chains

            except Exception as e:
                logger.debug(f"Could not load summary file {summary_file}: {e}")

        if chain_counts:
            logger.info(f"Loaded historical chain counts from {len(summary_files)} summary files")
            total_chains = sum(chain_counts.values())
            workflows_with_chains = len([w for w, c in chain_counts.items() if c > 0])
            logger.info(f"  Total historical chains: {total_chains} across {workflows_with_chains} workflows")

        return chain_counts

    def _save_cache_if_needed(self, force: bool = False):
        """
        Conditionally save cache based on time elapsed or workflow count.

        Args:
            force: If True, save immediately regardless of intervals
        """
        current_time = time.time()
        time_since_save = current_time - self.last_cache_save_time

        should_save = (
            force or
            time_since_save >= self.cache_save_interval or
            self.workflows_since_last_save >= self.cache_save_workflow_interval
        )

        if should_save:
            with self.cache_lock:
                logger.info(f"💾 Saving cache... (workflows since last save: {self.workflows_since_last_save}, time: {time_since_save:.0f}s)")
                save_start = time.time()
                self.booking_api.save_cache()
                save_duration = time.time() - save_start
                logger.info(f"✓ Cache saved in {save_duration:.2f}s")

                self.last_cache_save_time = current_time
                self.workflows_since_last_save = 0

    def get_existing_params(self, func_name: str) -> List[str]:
        """Extract existing parameter combinations for a function from cache.

        Returns:
            List of cache keys (strings) for this function
        """
        existing_params = []

        for cache_key in self.existing_cache.keys():
            if cache_key.startswith(f"{func_name}("):
                # Parse parameters from cache key
                # Example: "Search_Hotels(dest_id=123, arrival_date='2024-01-15', ...)"
                try:
                    params_str = cache_key[len(func_name)+1:-1]  # Remove "func(" and ")"
                    # This is a simplification - real parsing would be more complex
                    # For now, we'll just track that this combination exists
                    existing_params.append(cache_key)
                except Exception as e:
                    logger.debug(f"Could not parse cache key: {cache_key}")

        return existing_params

    def _get_function_specific_guidelines(self, func_name: str) -> str:
        """
        Get function-specific guidelines for LLM parameter generation.

        Based on empirical analysis of API success rates, certain functions
        have specific parameter constraints that improve success rates.

        Args:
            func_name: Function name

        Returns:
            Additional guidelines string (empty if no special guidelines needed)
        """
        # Dynamically calculate date range based on current date
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        start_month = (today + timedelta(days=30)).strftime("%Y-%m")
        end_month = (today + timedelta(days=180)).strftime("%Y-%m")
        date_range = f"{start_month} to {end_month}"

        # Example dates within the valid range
        example_date1 = (today + timedelta(days=45)).strftime("%Y-%m-%d")
        example_date2 = (today + timedelta(days=90)).strftime("%Y-%m-%d")
        example_date3 = (today + timedelta(days=120)).strftime("%Y-%m-%d")

        guidelines = {
            'Search_Car_Location': """
IMPORTANT for Search_Car_Location:
- Query must be a SPECIFIC car rental pickup location
- Use AIRPORT names (e.g., "Los Angeles International Airport", "Sydney Kingsford Smith Airport", "Paris - Charles de Gaulle Airport", "Melbourne Airport")
- Or use HOTEL names (e.g., "San Diego Marriott La Jolla", "JW Marriott Houston Downtown", "Loews Miami Beach Hotel")
- Or use "Downtown [City]" format (e.g., "Downtown Los Angeles", "Downtown San Francisco", "Downtown Boston")
- Or use city names (e.g., "San Diego", "Geneva", "Berlin, Germany")
- Locations should be globally diverse: US airports, European airports, Australian airports, Asian airports
""",
            'Search_Car_Rentals': f"""
IMPORTANT for Search_Car_Rentals:
- TODAY is {today_str}. All dates MUST be in the future!
- pick_up_date and drop_off_date: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- Format: YYYY-MM-DD
- Times MUST be on the hour: 08:00, 09:00, 10:00, 11:00, 12:00, 14:00, 15:00, 16:00 (NOT 08:15 or 10:30)
- Rental duration should typically be 1-7 days (drop_off_date - pick_up_date)
- Use coordinates from major airports or city centers worldwide
""",
            'Search_Taxi': f"""
IMPORTANT for Search_Taxi:
- TODAY is {today_str}. All dates MUST be in the future!
- pick_up_date: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- Format: YYYY-MM-DD
- Times MUST be on the hour: 08:00, 09:00, 10:00, 14:00, 15:00, 16:00
""",
            'Search_Flights': f"""
IMPORTANT for Search_Flights:
- TODAY is {today_str}. All dates MUST be in the future!
- departDate: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- returnDate is optional (omit for one-way). If provided, must be after departDate
- Format: YYYY-MM-DD
- children is ages comma-separated (e.g., "0,1,17" for infant, 1yo, 17yo). Use "" for no children
- adults: 1-4
""",
            'Search_Flights_Multi_Stops': f"""
IMPORTANT for Search_Flights_Multi_Stops:
- TODAY is {today_str}. All dates in legs MUST be in the future!
- Each leg's date: use dates from {date_range}
- Format: YYYY-MM-DD (e.g., {example_date1}, {example_date2}, {example_date3})
- Leg dates should be sequential (each leg's date >= previous leg's date)
- children is ages comma-separated (e.g., "0,1,17"). Omit or use "" for no children
""",
            'Get_Min_Price': f"""
IMPORTANT for Get_Min_Price:
- TODAY is {today_str}. All dates MUST be in the future!
- departDate: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- returnDate is optional. If provided, must be after departDate
- Format: YYYY-MM-DD
""",
            'Get_Min_Price_Multi_Stops': f"""
IMPORTANT for Get_Min_Price_Multi_Stops:
- TODAY is {today_str}. All dates in legs MUST be in the future!
- Each leg's date: use dates from {date_range}
- Format: YYYY-MM-DD (e.g., {example_date1}, {example_date2}, {example_date3})
""",
            'Search_Hotels': f"""
IMPORTANT for Search_Hotels:
- TODAY is {today_str}. All dates MUST be in the future!
- arrival_date: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- departure_date: must be after arrival_date (typically 1-14 days later)
- Format: YYYY-MM-DD
- adults: MUST be at least 1 (never use 0)
""",
            'Search_Hotels_By_Coordinates': f"""
IMPORTANT for Search_Hotels_By_Coordinates:
- TODAY is {today_str}. All dates MUST be in the future!
- arrival_date: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- departure_date: must be after arrival_date (typically 1-14 days later)
- Format: YYYY-MM-DD
- adults: MUST be at least 1 (never use 0)
""",
            'Get_Filter': f"""
IMPORTANT for Get_Filter:
- TODAY is {today_str}. All dates MUST be in the future!
- arrival_date: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- departure_date: must be after arrival_date
- Format: YYYY-MM-DD
""",
            'Get_Room_Availability': f"""
IMPORTANT for Get_Room_Availability:
- TODAY is {today_str}. All dates MUST be in the future!
- min_date and max_date: use dates from {date_range}
- max_date must be after min_date
- Format: YYYY-MM-DD
""",
            'Get_Room_List_With_Availability': f"""
IMPORTANT for Get_Room_List_With_Availability:
- TODAY is {today_str}. All dates MUST be in the future!
- arrival_date: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- departure_date: must be after arrival_date
- Format: YYYY-MM-DD
- adults: MUST be at least 1 (never use 0)
""",
            'Get_Availability': f"""
IMPORTANT for Get_Availability:
- TODAY is {today_str}. The date MUST be in the future!
- date: use dates from {date_range} (e.g., {example_date1}, {example_date2})
- Format: YYYY-MM-DD
""",
            'Search_Attractions': """
IMPORTANT for Search_Attractions:
- sortBy must be exactly one of: trending, lowest_price, attr_book_score (lowercase only, single value)
""",
        }
        return guidelines.get(func_name, '')

    def generate_diverse_params(
        self,
        func_name: str,
        target_count: int,
        existing_params: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to generate diverse parameter combinations for a function.

        Args:
            func_name: Function name
            target_count: Number of new parameter combinations to generate
            existing_params: Existing cache keys to avoid duplicates

        Returns:
            List of parameter dictionaries
        """
        if func_name not in self.tool_schemas:
            logger.warning(f"Function {func_name} not found in tool schemas")
            return []

        func_schema = self.tool_schemas[func_name]

        # Build system message
        system_msg = (
            f"You are generating diverse parameter combinations for the {func_name} function. "
            "Generate realistic, varied parameter values that would be useful for testing. "
            "\n\nIMPORTANT:\n"
            "- Generate realistic values (real city names, valid dates, etc.)\n"
            "- Maximize diversity (different locations, date ranges, options)\n"
            "- Return ONLY parameter values, no natural language queries\n"
            "- Follow the function schema exactly"
        )

        # Extract parameter schema
        parameters = func_schema.get('parameters', {})
        required = func_schema.get('required', [])

        # Validate schema structure
        if not isinstance(parameters, dict):
            logger.warning(f"Invalid parameters schema for {func_name} (expected dict, got {type(parameters).__name__})")
            return []

        params_description = f"Function: {func_name}\n\nParameters:\n"
        for param_name, param_schema in parameters.items():
            param_type = param_schema.get('type', 'unknown')
            param_desc = param_schema.get('description', '')
            required_marker = " (REQUIRED)" if param_name in required else " (optional)"
            params_description += f"  - {param_name} ({param_type}){required_marker}: {param_desc}\n"

        # Build user message
        user_msg = f"""
{params_description}

Existing cache entries: {len(existing_params)} combinations already cached

Task: Generate {target_count} NEW diverse parameter combinations for this function.

Guidelines:
- Focus on diversity: vary locations, dates, numeric values, enums
- Use realistic values (real cities, valid date formats, reasonable numbers)
- For dates: use format YYYY-MM-DD, vary across different months/years
- For location IDs: use a wide range of values
- For enums: try all available options
- Avoid duplicating existing combinations
{self._get_function_specific_guidelines(func_name)}
OUTPUT FORMAT (JSON):
{{
  "parameter_sets": [
    {{
      "param1": value1,
      "param2": value2,
      ...
    }},
    ...
  ]
}}

Generate {target_count} diverse parameter combinations now.
"""

        try:
            prompt = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]

            response = get_llm_response(prompt, model=self.model, return_type="text_only")
            parsed = parse_json(response)

            if 'parameter_sets' not in parsed:
                logger.warning(f"LLM response missing 'parameter_sets' field for {func_name}")
                return []

            param_sets = parsed['parameter_sets']
            logger.info(f"LLM generated {len(param_sets)} parameter combinations for {func_name}")

            return param_sets

        except Exception as e:
            logger.error(f"Error generating parameters for {func_name}: {e}")
            return []

    def execute_and_cache(self, func_name: str, params: Dict[str, Any]) -> bool:
        """
        Execute a function with given parameters and cache the result.

        Args:
            func_name: Function name
            params: Parameter dictionary

        Returns:
            True if execution succeeded and was cached, False otherwise
        """
        self.expansion_stats['attempts'][func_name] += 1

        try:
            # Get the method from BookingAPI
            if not hasattr(self.booking_api, func_name):
                logger.warning(f"Function {func_name} not found in BookingAPI")
                self.expansion_stats['failures'][func_name] += 1
                return False

            method = getattr(self.booking_api, func_name)

            # Execute function
            logger.debug(f"Executing {func_name} with params: {params}")
            result = method(**params)

            # Check if result is an error
            if isinstance(result, dict) and 'error' in result:
                logger.debug(f"API returned error for {func_name}: {result['error']}")
                self.expansion_stats['failures'][func_name] += 1
                return False

            # Success! Cache was automatically updated by BookingAPI
            self.expansion_stats['successes'][func_name] += 1
            logger.info(f"✓ Successfully cached {func_name} with new parameters")
            return True

        except Exception as e:
            logger.error(f"Error executing {func_name}: {e}")
            self.expansion_stats['failures'][func_name] += 1
            return False

    def expand_function_cache(self, func_name: str, target_samples: int) -> int:
        """
        Expand cache for a specific function.

        Args:
            func_name: Function name to expand
            target_samples: Number of new samples to attempt

        Returns:
            Number of successful new cache entries
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Expanding cache for: {func_name}")
        logger.info(f"Target: {target_samples} new samples")
        logger.info(f"Existing: {self.cache_stats[func_name]} cached entries")
        logger.info(f"{'='*60}")

        # Get existing parameter combinations
        existing_params = self.get_existing_params(func_name)

        # Generate diverse parameter combinations
        param_sets = self.generate_diverse_params(func_name, target_samples, existing_params)

        if not param_sets:
            logger.warning(f"No parameter sets generated for {func_name}")
            return 0

        # Execute each parameter combination
        new_entries = 0
        for i, params in enumerate(param_sets, 1):
            logger.info(f"[{i}/{len(param_sets)}] Testing {func_name} with generated params...")

            if self.execute_and_cache(func_name, params):
                new_entries += 1
                self.expansion_stats['new_entries'][func_name] += 1

        logger.info(f"\n{func_name} expansion complete: {new_entries}/{len(param_sets)} successful")
        return new_entries

    def expand_cache(
        self,
        functions: List[str],
        samples_per_function: int
    ) -> Dict[str, int]:
        """
        Expand cache for multiple functions.

        Args:
            functions: List of function names (or ['all'])
            samples_per_function: Target samples per function

        Returns:
            Dictionary mapping function names to new entry counts
        """
        # Handle "all" functions
        if 'all' in functions:
            functions = list(self.tool_schemas.keys())
            logger.info(f"Expanding cache for ALL {len(functions)} functions")

        results = {}
        total_new_entries = 0

        for func_name in functions:
            new_entries = self.expand_function_cache(func_name, samples_per_function)
            results[func_name] = new_entries
            total_new_entries += new_entries

        # Print final statistics
        logger.info(f"\n{'='*60}")
        logger.info("CACHE EXPANSION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Successful API calls: {total_new_entries}")
        logger.info(f"Functions expanded: {len(results)}")
        logger.info(f"\nDetailed statistics:")

        for func_name in sorted(results.keys()):
            attempts = self.expansion_stats['attempts'][func_name]
            successes = self.expansion_stats['successes'][func_name]
            failures = self.expansion_stats['failures'][func_name]
            success_rate = (successes / attempts * 100) if attempts > 0 else 0

            logger.info(
                f"  {func_name}: {successes}/{attempts} successful ({success_rate:.1f}%)"
            )

        # Reload cache to get actual final count
        logger.info(f"\nCache file: {self.cache_path}")
        try:
            with open(self.cache_path) as f:
                final_cache = json.load(f)
            actual_growth = len(final_cache) - len(self.existing_cache)
            logger.info(f"Cache growth: {len(self.existing_cache)} → {len(final_cache)} entries (+{actual_growth})")

            if actual_growth != total_new_entries:
                logger.info(f"Note: {total_new_entries - actual_growth} API calls were duplicates (already in cache)")
        except Exception as e:
            logger.warning(f"Could not reload cache for final count: {e}")

        return results

    def extract_field_value(self, response: Dict[str, Any], field_path: str) -> Any:
        """
        Extract value from nested API response using path notation.

        Handles both response formats:
        1. With 'data' wrapper: {status, message, data: {destinations: [...]}}
        2. Without 'data' wrapper: {destinations: [...], products: [...]}

        Args:
            response: API response dict
            field_path: Path like "destinations[0].id" or "coordinates.latitude"

        Returns:
            Extracted value or None if path invalid
        """
        import re

        def _extract(current, tokens):
            """Helper to extract value from tokens."""
            try:
                for key, index in tokens:
                    if index:
                        current = current[int(index)]
                    elif key:
                        if isinstance(current, dict):
                            if key not in current:
                                return None
                            current = current[key]
                        else:
                            return None
                return current
            except (KeyError, IndexError, TypeError, AttributeError):
                return None

        try:
            # Parse field path into tokens
            tokens = re.findall(r'(\w+)|\[(\d+)\]', field_path)

            # Try 1: With 'data' wrapper
            if isinstance(response, dict) and 'data' in response:
                if tokens and tokens[0][0] != 'data':
                    result = _extract(response['data'], tokens)
                    if result is not None:
                        return result

            # Try 2: Without 'data' wrapper (direct access)
            result = _extract(response, tokens)
            if result is not None:
                return result

            return None

        except Exception:
            return None

    def _is_valid_response_for_dependency(self, response: Any, field_path: Optional[str] = None) -> bool:
        """
        Check if a response contains data that can be used for dependency extraction.

        Args:
            response: API response
            field_path: Optional specific field path to check (e.g., 'destinations[0].id')
                       If provided, validates that path is extractable.
                       If not provided, checks if any result field has data.

        Returns:
            False if response has empty result arrays or cannot extract the specific field.
        """
        if not isinstance(response, dict):
            # List responses are valid if non-empty
            return isinstance(response, list) and len(response) > 0

        # If specific field path provided, check if we can extract it
        if field_path:
            # Extract the root field (e.g., 'destinations' from 'destinations[0].id')
            import re
            match = re.match(r'(\w+)', field_path)
            if match:
                root_field = match.group(1)

                # Try with 'data' wrapper
                data = response.get('data', response)

                if root_field in data:
                    value = data[root_field]
                    # Check if it's a non-empty array (for paths like destinations[0].id)
                    if isinstance(value, list):
                        return len(value) > 0
                    # Or a non-empty dict
                    elif isinstance(value, dict):
                        return len(value) > 0

                return False

        # No specific path - check if ANY field in data has non-empty list/dict
        # This handles all API response formats without hardcoding field names
        data = response.get('data', response)

        if not isinstance(data, dict):
            # data itself is a list (e.g., Search_Car_Location, Search_Flight_Location)
            return isinstance(data, list) and len(data) > 0

        # Check all fields in data for non-empty list or dict values
        for field, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                return True
            elif isinstance(value, dict) and len(value) > 0:
                # For dict values, check if it contains meaningful data (not just metadata)
                # Skip common metadata fields that don't indicate valid results
                if field not in ('meta', 'search_context', 'pagination'):
                    return True

        return False

    def _execute_step(self, func_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a function step and return the response (caching happens automatically).
        Thread-safe for parallel execution.

        Args:
            func_name: Function name
            params: Parameter dictionary

        Returns:
            Response dict or None if failed
        """
        try:
            # Check for shutdown request
            if self._shutdown_requested:
                return None

            if not hasattr(self.booking_api, func_name):
                logger.warning(f"Function {func_name} not found in BookingAPI")
                return None

            method = getattr(self.booking_api, func_name)

            # Filter params to only include valid function parameters
            # This handles cases where LLM generates invalid parameter names
            sig = inspect.signature(method)
            valid_param_names = set(sig.parameters.keys())
            filtered_params = {k: v for k, v in params.items() if k in valid_param_names}

            if len(filtered_params) < len(params):
                invalid_params = set(params.keys()) - valid_param_names
                logger.debug(f"Filtered out invalid params for {func_name}: {invalid_params}")

            # Type conversion: BookingAPI expects certain parameters as strings
            # hotel_id: LLM often generates int, but API expects string
            if 'hotel_id' in filtered_params and not isinstance(filtered_params['hotel_id'], str):
                filtered_params['hotel_id'] = str(filtered_params['hotel_id'])

            # latitude/longitude: LLM often generates float, but API expects string
            for coord_param in ['latitude', 'longitude']:
                if coord_param in filtered_params and not isinstance(filtered_params[coord_param], str):
                    filtered_params[coord_param] = str(filtered_params[coord_param])

            # Check if this would be a cache hit (before calling)
            with self.cache_lock:
                current_cache_size = len(self.booking_api._cache.get(func_name, []))

            # Execute function (automatically caches on success)
            # NOTE: API call is outside the lock to allow parallel execution.
            # BookingAPI's internal cache operations may have minor race conditions
            # (worst case: duplicate cache entries), but this is acceptable for
            # cache expansion where parallelism is critical for performance.
            logger.debug(f"Executing {func_name} with params: {filtered_params}")

            # Track API call timing - call is OUTSIDE the lock for parallelism
            api_start = time.time()
            result = method(**filtered_params)  # No lock here - allows parallel API calls
            api_duration = time.time() - api_start

            # Record timing (lock only for statistics update)
            with self.cache_lock:
                self.expansion_stats['api_timings'].append(api_duration)

            # Log slow API calls
            if api_duration > 2.0:
                logger.warning(f"Slow API call: {func_name} took {api_duration:.2f}s")

            # Check if cache grew (new entry) or stayed same (cache hit)
            with self.cache_lock:
                new_cache_size = len(self.booking_api._cache.get(func_name, []))
                if new_cache_size == current_cache_size and result and 'error' not in str(result):
                    self.expansion_stats['cache_hits'][func_name] += 1
                    logger.debug(f"  ✓ Cache hit for {func_name}")

            # Check for errors - multiple error patterns:
            # 1. Response has 'error' key
            # 2. Response has 'status': false (API error without data)
            # 3. Response is server-side error (should be treated as failure)
            if isinstance(result, dict):
                if 'error' in result:
                    logger.debug(f"API returned error: {result['error']}")
                    return None
                if result.get('status') is False and 'data' not in result:
                    logger.debug(f"API returned status=false: {result.get('message', 'unknown error')}")
                    return None
                # Check for server-side errors that shouldn't be used
                message = result.get('message', '')
                server_error_patterns = ["Something went wrong", "Server down", "<!doctype html>", "Not Found"]
                if any(pattern in message for pattern in server_error_patterns):
                    logger.debug(f"API returned server error: {message[:100]}")
                    return None

            # Also reject empty list responses
            if isinstance(result, list) and len(result) == 0:
                logger.debug(f"API returned empty list response")
                return None

            return result

        except Exception as e:
            logger.error(f"Error executing {func_name}: {e}")
            return None

    def _generate_step_params_batch(
        self,
        func_name: str,
        dep_info: Dict[str, Any],
        dependent_params: Optional[Dict[str, Any]] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Batch generate multiple parameter combinations for a step using LLM.

        Args:
            func_name: Function name
            dep_info: Dependency info from workflow template
            dependent_params: Parameters extracted from previous steps (if any)
            batch_size: Number of parameter combinations to generate

        Returns:
            List of parameter dicts
        """
        func_schema = self.tool_schemas.get(func_name)
        if not func_schema:
            logger.warning(f"No schema found for {func_name}")
            return []

        # Identify which parameters need LLM generation
        parameters_schema = func_schema.get('parameters', {})
        all_params = parameters_schema.get('properties', {})
        required_params = parameters_schema.get('required', [])

        dependent_param_names = set(dependent_params.keys()) if dependent_params else set()
        independent_params = [
            p for p in all_params.keys()
            if p not in dependent_param_names
        ]

        if not independent_params:
            # All params are dependent - return single combination
            return [dependent_params] if dependent_params else []

        # Build parameter description
        params_desc = f"Function: {func_name}\n\nGenerate values for:\n"
        for param in independent_params:
            param_schema = all_params[param]
            param_type = param_schema.get('type', 'unknown')
            param_desc = param_schema.get('description', '')
            required_marker = " (REQUIRED)" if param in required_params else " (optional)"
            params_desc += f"  - {param} ({param_type}){required_marker}: {param_desc}\n"

        if dependent_params:
            params_desc += f"\nThese parameters are automatically filled from previous step:\n"
            for dep_param, dep_value in dependent_params.items():
                params_desc += f"  - {dep_param} = {dep_value}\n"

        # Generate parameters via LLM
        system_msg = (
            f"You are generating diverse, realistic parameter values for the {func_name} function. "
            "Generate parameter combinations that would be useful for API testing. "
            "Maximize diversity across different parameter values."
        )

        user_msg = f"""
{params_desc}

Task: Generate {batch_size} DIVERSE parameter combinations.

Requirements:
- Maximize diversity (different locations, dates, values)
- All combinations should be realistic and valid
- Use different values for each combination
{self._get_function_specific_guidelines(func_name)}
OUTPUT FORMAT (JSON):
{{
  "combinations": [
    {{"param1": value1, "param2": value2, ...}},
    {{"param1": value1_alt, "param2": value2_alt, ...}},
    ...  // {batch_size} combinations total
  ]
}}

Generate {batch_size} diverse parameter combinations now.
"""

        try:
            prompt = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]

            response = get_llm_response(prompt, model=self.model, return_type="text_only")
            parsed = parse_json(response)

            if 'combinations' not in parsed:
                logger.warning(f"LLM response missing 'combinations' field for {func_name}")
                return []

            candidates = parsed.get('combinations', [])

            # Merge with dependent params if provided
            if dependent_params:
                candidates = [
                    {**candidate, **dependent_params}
                    for candidate in candidates
                ]

            logger.info(f"Generated {len(candidates)} parameter combinations for {func_name}")
            return candidates

        except Exception as e:
            logger.error(f"Error generating parameters for {func_name}: {e}")
            return []

    def generate_workflow_chain(
        self,
        workflow: Dict[str, Any],
        batch_size_step0: int = 20,
        batch_size_dependent: int = 3,
        max_branches_per_step: int = 60,
        max_workers: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate complete workflow chains using batch generation + tree expansion.

        Strategy:
        1. For independent steps: Generate N candidates, validate all in parallel
        2. For dependent steps: For each successful previous result, generate K candidates in parallel
        3. Build tree of valid candidates level-by-level
        4. Apply branch pruning to prevent exponential growth
        5. Return ALL complete chains (all successful paths are cached)

        Args:
            workflow: Workflow definition from workflow_templates.json
            batch_size_step0: Batch size for independent steps
            batch_size_dependent: Batch size for dependent steps (per branch)
            max_branches_per_step: Maximum number of candidates to keep per step (prevents exponential growth)
            max_workers: Number of parallel workers for API calls (default: 10)

        Returns:
            List of ALL successful workflow chains
        """
        logger.info(f"\nGenerating workflow chains for: {workflow['id']}")
        logger.info(f"Pattern: {' → '.join(workflow['pattern'])}")
        logger.info(f"Branch limit: {max_branches_per_step} candidates per step")
        logger.info(f"Parallel workers: {max_workers}")

        # Tree structure: step_idx -> list of candidates
        # Each candidate: {'params': ..., 'response': ..., 'path': [...]}
        tree = {}

        # Build tree level by level
        for step_idx_str, dep_info in sorted(workflow['dependencies'].items(),
                                             key=lambda x: int(x[0])):
            # Check for shutdown at the start of each step
            if self._shutdown_requested:
                logger.warning(f"Shutdown requested, stopping workflow {workflow['id']}")
                break

            step_idx = int(step_idx_str)
            func_name = dep_info['function']

            logger.info(f"\nStep {step_idx}: {func_name}")

            tree[step_idx] = []

            if not dep_info['depends_on']:
                # Independent step - generate batch
                logger.info(f"  Generating {batch_size_step0} candidates (independent step)")
                candidates = self._generate_step_params_batch(
                    func_name=func_name,
                    dep_info=dep_info,
                    batch_size=batch_size_step0
                )

                # For Step 0, check what fields the next step needs
                next_step_field_paths = None
                if step_idx + 1 < len(workflow['pattern']):
                    # Get dependency info for next step
                    next_step_key = str(step_idx + 1)
                    if next_step_key in workflow['dependencies']:
                        next_dep_info = workflow['dependencies'][next_step_key]
                        if 'dependency_args' in next_dep_info and next_dep_info['dependency_args']:
                            # Extract field paths that next step will need
                            next_step_field_paths = [
                                dep_detail['from_field']
                                for dep_detail in next_dep_info['dependency_args'].values()
                            ]
                            logger.debug(f"  Next step will need fields: {next_step_field_paths}")

                # Validate all candidates in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_params = {
                        executor.submit(self._execute_step, func_name, params): (i, params)
                        for i, params in enumerate(candidates, 1)
                    }

                    # Process results as they complete
                    for future in as_completed(future_to_params):
                        # Check for shutdown
                        if self._shutdown_requested:
                            logger.warning("  Shutdown requested, cancelling pending tasks...")
                            for f in future_to_params:
                                f.cancel()
                            break

                        i, params = future_to_params[future]
                        try:
                            response = future.result()
                            if response and 'error' not in str(response):
                                # Check if response contains data needed for next step
                                is_valid = True
                                if next_step_field_paths:
                                    # Validate each required field
                                    for field_path in next_step_field_paths:
                                        if not self._is_valid_response_for_dependency(response, field_path):
                                            is_valid = False
                                            logger.debug(f"  ⚠ Candidate {i}/{len(candidates)} succeeded but missing field for next step: {field_path}")
                                            break
                                else:
                                    # No next step, just check if has any valid data
                                    is_valid = self._is_valid_response_for_dependency(response)

                                if is_valid:
                                    tree[step_idx].append({
                                        'params': params,
                                        'response': response,
                                        'path': []  # No previous steps
                                    })
                                    logger.debug(f"  ✓ Candidate {i}/{len(candidates)} succeeded (valid data)")
                                else:
                                    if not next_step_field_paths:
                                        logger.debug(f"  ⚠ Candidate {i}/{len(candidates)} succeeded but returned empty results")
                            else:
                                logger.debug(f"  ✗ Candidate {i}/{len(candidates)} failed")
                        except Exception as e:
                            logger.error(f"  ✗ Candidate {i}/{len(candidates)} raised exception: {e}")

                # Apply branch pruning for Step 0 (random sampling for diversity)
                if len(tree[step_idx]) > max_branches_per_step:
                    logger.info(f"  Result: {len(tree[step_idx])}/{len(candidates)} successful → randomly sampled {max_branches_per_step}")
                    tree[step_idx] = random.sample(tree[step_idx], max_branches_per_step)
                else:
                    logger.info(f"  Result: {len(tree[step_idx])}/{len(candidates)} successful")

            else:
                # Dependent step - need to gather responses from all dependent steps
                depends_on_steps = dep_info['depends_on']

                # Check all dependent steps have candidates
                missing_steps = [s for s in depends_on_steps if s not in tree or not tree[s]]
                if missing_steps:
                    logger.warning(f"  No candidates from steps {missing_steps} - skipping")
                    continue

                # Build combinations of candidates from dependent steps
                # For multi-step dependencies (e.g., step 2 depends on [0, 1]),
                # we need to create combinations from independent branches

                # Get unique from_steps needed for this step's dependency args
                needed_steps = set(
                    dep_detail['from_step']
                    for dep_detail in dep_info['dependency_args'].values()
                )

                # Create combinations of candidates from needed steps
                step_candidates = [tree[s] for s in sorted(needed_steps)]
                if len(step_candidates) == 1:
                    # Single dependency - simple case
                    combinations = [(c,) for c in step_candidates[0]]
                else:
                    # Multiple dependencies - create combinations (limit to prevent explosion)
                    max_combinations = max_branches_per_step * 2
                    all_combos = list(product(*step_candidates))
                    if len(all_combos) > max_combinations:
                        combinations = random.sample(all_combos, max_combinations)
                        logger.info(f"  Sampled {max_combinations} combinations from {len(all_combos)} possible")
                    else:
                        combinations = all_combos

                logger.info(f"  Processing {len(combinations)} candidate combinations from steps {sorted(needed_steps)}")

                # Process all branches in parallel
                def process_branch(branch_data):
                    """Process a single branch: extract deps, generate params, execute API calls."""
                    branch_idx, candidate_combo = branch_data

                    # Build step_responses map from the combination
                    step_responses = {}
                    sorted_steps = sorted(needed_steps)
                    for i, candidate in enumerate(candidate_combo):
                        step_responses[sorted_steps[i]] = candidate['response']

                    # First check if all required responses have required fields
                    has_all_fields = True
                    for arg_name, dep_detail in dep_info['dependency_args'].items():
                        from_step = dep_detail['from_step']
                        field_path = dep_detail['from_field']

                        if from_step not in step_responses:
                            logger.debug(f"  [Branch {branch_idx}] ⚠ Skipping: Missing response for step {from_step}")
                            has_all_fields = False
                            break

                        if not self._is_valid_response_for_dependency(step_responses[from_step], field_path):
                            logger.debug(f"  [Branch {branch_idx}] ⚠ Skipping: Step {from_step} response missing field: {field_path}")
                            has_all_fields = False
                            break

                    if not has_all_fields:
                        return {'branch_idx': branch_idx, 'generated': 0, 'results': []}

                    # Extract dependent params from the CORRECT step's response
                    dependent_params = {}
                    for arg_name, dep_detail in dep_info['dependency_args'].items():
                        from_step = dep_detail['from_step']
                        field_path = dep_detail['from_field']
                        value = self.extract_field_value(step_responses[from_step], field_path)
                        if value is not None:
                            dependent_params[arg_name] = value

                    if not dependent_params:
                        logger.debug(f"  [Branch {branch_idx}] ⚠ Skipping: Could not extract any dependent params")
                        return {'branch_idx': branch_idx, 'generated': 0, 'results': []}

                    # Generate batch for this branch (LLM call)
                    candidates = self._generate_step_params_batch(
                        func_name=func_name,
                        dep_info=dep_info,
                        dependent_params=dependent_params,
                        batch_size=batch_size_dependent
                    )

                    # Execute API calls for this branch in parallel (within the branch)
                    branch_results = []
                    with ThreadPoolExecutor(max_workers=len(candidates)) as branch_executor:
                        future_to_params = {
                            branch_executor.submit(self._execute_step, func_name, params): params
                            for params in candidates
                        }

                        for future in as_completed(future_to_params):
                            # Check for shutdown
                            if self._shutdown_requested:
                                for f in future_to_params:
                                    f.cancel()
                                break

                            params = future_to_params[future]
                            try:
                                response = future.result()
                                if response and 'error' not in str(response):
                                    if self._is_valid_response_for_dependency(response):
                                        # Build cumulative path from all source candidates and their ancestors
                                        # This ensures the full chain can be reconstructed later
                                        merged_path = []
                                        seen_ids = set()
                                        for cand in candidate_combo:
                                            # Add ancestors from this candidate's path
                                            for ancestor in cand.get('path', []):
                                                if id(ancestor) not in seen_ids:
                                                    merged_path.append(ancestor)
                                                    seen_ids.add(id(ancestor))
                                            # Add the candidate itself
                                            if id(cand) not in seen_ids:
                                                merged_path.append(cand)
                                                seen_ids.add(id(cand))

                                        branch_results.append({
                                            'params': params,
                                            'response': response,
                                            'path': merged_path
                                        })
                            except Exception as e:
                                logger.debug(f"  API call for {params} failed: {e}")

                    return {
                        'branch_idx': branch_idx,
                        'generated': len(candidates),
                        'results': branch_results
                    }

                # Process all branches in parallel
                total_generated = 0
                total_succeeded = 0

                branch_data = list(enumerate(combinations, 1))

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all branch processing tasks
                    future_to_branch = {
                        executor.submit(process_branch, (idx, candidate)): idx
                        for idx, candidate in branch_data
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_branch):
                        # Check for shutdown
                        if self._shutdown_requested:
                            logger.warning("  Shutdown requested, cancelling pending tasks...")
                            for f in future_to_branch:
                                f.cancel()
                            break

                        branch_idx = future_to_branch[future]
                        try:
                            result = future.result()
                            total_generated += result['generated']
                            total_succeeded += len(result['results'])

                            # Add successful results to tree
                            tree[step_idx].extend(result['results'])

                            if result['generated'] > 0:
                                logger.info(f"  [Branch {result['branch_idx']}] {len(result['results'])}/{result['generated']} successful")
                        except Exception as e:
                            logger.error(f"  [Branch {branch_idx}] Failed: {e}")

                # Apply branch pruning for dependent steps (random sampling for diversity)
                original_count = len(tree[step_idx])
                if original_count > max_branches_per_step:
                    logger.info(f"  Result: {original_count}/{total_generated} successful → randomly sampled {max_branches_per_step}")
                    tree[step_idx] = random.sample(tree[step_idx], max_branches_per_step)
                else:
                    logger.info(f"  Result: {original_count}/{total_generated} successful")

        # Extract complete chains from final level
        final_step = max(tree.keys()) if tree else -1
        if final_step == -1 or not tree[final_step]:
            logger.warning(f"No complete chains generated for {workflow['id']}")
            return []

        successful_chains = []
        for final_candidate in tree[final_step]:
            # Reconstruct full chain
            chain_steps = []
            for step_node in final_candidate['path'] + [final_candidate]:
                # Determine which step this is
                for idx, dep_info in workflow['dependencies'].items():
                    if dep_info['function'] == step_node.get('function'):
                        step_func = dep_info['function']
                        break
                else:
                    # Find function from params
                    step_func = None
                    for func in self.tool_schemas.keys():
                        if hasattr(self.booking_api, func):
                            step_func = func
                            break

                # Note: We need to track function name in tree nodes
                # For now, reconstruct from workflow dependencies
                pass

            # Simpler approach: reconstruct from workflow dependencies
            full_path = final_candidate['path'] + [final_candidate]
            chain_steps = []
            for i, (step_idx_str, dep_info) in enumerate(sorted(workflow['dependencies'].items(),
                                                                 key=lambda x: int(x[0]))):
                if i < len(full_path):
                    node = full_path[i]
                    chain_steps.append({
                        'step': int(step_idx_str),
                        'function': dep_info['function'],
                        'arguments': node['params'],
                        'response': node['response']
                    })

            successful_chains.append({
                'workflow_id': workflow['id'],
                'steps': chain_steps
            })

        logger.info(f"\n✓ Generated {len(successful_chains)} complete chains for {workflow['id']}")
        return successful_chains

    def expand_workflow_cache(
        self,
        workflow_ids: Optional[List[str]] = None,
        exclude_workflow_ids: Optional[List[str]] = None,
        batch_size_step0: int = 20,
        batch_size_dependent: int = 3,
        max_branches_per_step: int = 60,
        max_workers: int = 10,
        workflow_workers: int = 1,
        max_chains_per_workflow: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Expand cache by generating complete workflow chains.

        Args:
            workflow_ids: List of workflow IDs to expand (None = all workflows)
            exclude_workflow_ids: List of workflow IDs to exclude (skip)
            batch_size_step0: Batch size for independent steps
            batch_size_dependent: Batch size for dependent steps
            max_branches_per_step: Maximum branches per step (prevents exponential growth)
            max_workers: Number of parallel workers for API calls within each workflow
            workflow_workers: Number of workflows to process in parallel
            max_chains_per_workflow: Maximum chains per workflow (skip if already has >= this many)

        Returns:
            Dict mapping workflow_id to number of successful chains generated
        """
        if self.workflows is None:
            raise RuntimeError(
                "Workflow templates not loaded. "
                "Initialize CacheExpander with workflow_templates_path parameter."
            )

        # Select workflows to expand
        if workflow_ids is None or 'all' in workflow_ids:
            workflows = self.workflows
        else:
            workflows = [w for w in self.workflows if w['id'] in workflow_ids]

        # Exclude specified workflows
        if exclude_workflow_ids:
            exclude_set = set(exclude_workflow_ids)
            workflows = [w for w in workflows if w['id'] not in exclude_set]
            logger.info(f"Excluded {len(exclude_workflow_ids)} workflows, {len(workflows)} remaining")

        # Exclude workflows that have already reached max_chains_per_workflow
        skipped_workflows = []
        if max_chains_per_workflow is not None:
            remaining_workflows = []
            for w in workflows:
                existing_chains = self.historical_chain_counts.get(w['id'], 0)
                if existing_chains >= max_chains_per_workflow:
                    skipped_workflows.append((w['id'], existing_chains))
                else:
                    remaining_workflows.append(w)
            workflows = remaining_workflows

            if skipped_workflows:
                logger.info(f"Skipping {len(skipped_workflows)} workflows (already >= {max_chains_per_workflow} chains):")
                for wf_id, count in skipped_workflows[:10]:
                    logger.info(f"  - {wf_id}: {count} chains")
                if len(skipped_workflows) > 10:
                    logger.info(f"  ... and {len(skipped_workflows) - 10} more")
                logger.info(f"{len(workflows)} workflows remaining to process")

        if not workflows:
            logger.error("No workflows selected for expansion")
            return {}

        # Get initial cache size
        initial_cache_size = sum(len(entries) for entries in self.booking_api._cache.values())

        results = {}
        total_chains = 0
        total_steps = 0
        workflow_stats = {}  # Detailed per-workflow statistics
        import time

        logger.info(f"\n{'='*70}")
        logger.info("WORKFLOW-AWARE CACHE EXPANSION (PARALLEL MODE)")
        logger.info(f"{'='*70}")
        logger.info(f"Workflows to expand: {len(workflows)}")
        if skipped_workflows:
            logger.info(f"Workflows skipped (at limit): {len(skipped_workflows)}")
        logger.info(f"Batch sizes: step0={batch_size_step0}, dependent={batch_size_dependent}")
        logger.info(f"Max branches per step: {max_branches_per_step} (prevents exponential growth)")
        if max_chains_per_workflow is not None:
            logger.info(f"Max chains per workflow: {max_chains_per_workflow}")
        logger.info(f"Parallel workers: {max_workers} (for API calls)")
        logger.info(f"Workflow workers: {workflow_workers} (parallel workflows)")
        logger.info(f"Initial cache size: {initial_cache_size} entries")
        logger.info(f"{'='*70}\n")

        # Process workflows in parallel
        def process_workflow(workflow):
            """Process a single workflow and return results."""
            workflow_id = workflow['id']
            workflow_start_time = time.time()

            chains = self.generate_workflow_chain(
                workflow,
                batch_size_step0=batch_size_step0,
                batch_size_dependent=batch_size_dependent,
                max_branches_per_step=max_branches_per_step,
                max_workers=max_workers
            )

            workflow_duration = time.time() - workflow_start_time
            num_chains = len(chains)
            num_steps = sum(len(chain['steps']) for chain in chains)

            return {
                'workflow_id': workflow_id,
                'num_chains': num_chains,
                'num_steps': num_steps,
                'duration': workflow_duration,
                'pattern': ' → '.join(workflow['pattern']),
                'num_steps_in_pattern': len(workflow['pattern'])
            }

        # Use ThreadPoolExecutor to process workflows in parallel
        if workflow_workers > 1:
            logger.info(f"Processing workflows in parallel with {workflow_workers} workers...\n")
            with ThreadPoolExecutor(max_workers=workflow_workers) as executor:
                future_to_workflow = {
                    executor.submit(process_workflow, wf): wf['id']
                    for wf in workflows
                }

                for future in as_completed(future_to_workflow):
                    # Check for shutdown
                    if self._shutdown_requested:
                        logger.warning("Shutdown requested, cancelling pending workflows...")
                        for f in future_to_workflow:
                            f.cancel()
                        break

                    workflow_id = future_to_workflow[future]
                    try:
                        result = future.result()
                        workflow_id = result['workflow_id']

                        results[workflow_id] = result['num_chains']
                        total_chains += result['num_chains']
                        total_steps += result['num_steps']

                        # Record detailed stats
                        workflow_stats[workflow_id] = {
                            'num_chains': result['num_chains'],
                            'num_steps_in_pattern': result['num_steps_in_pattern'],
                            'pattern': result['pattern'],
                            'duration_seconds': round(result['duration'], 2),
                            'status': 'success' if result['num_chains'] > 0 else 'failed'
                        }

                        status_symbol = '✓' if result['num_chains'] > 0 else '✗'
                        logger.info(f"{status_symbol} {workflow_id} completed: {result['num_chains']} chains in {result['duration']:.1f}s")

                        # Increment workflow counter and check if we should save
                        self.workflows_since_last_save += 1
                        self._save_cache_if_needed(force=False)

                    except Exception as e:
                        logger.error(f"✗ {workflow_id} failed with exception: {e}")
                        workflow_stats[workflow_id] = {
                            'num_chains': 0,
                            'num_steps_in_pattern': 0,
                            'pattern': '',
                            'duration_seconds': 0,
                            'status': 'failed'
                        }
        else:
            # Sequential processing (original behavior)
            for workflow in workflows:
                # Check for shutdown
                if self._shutdown_requested:
                    logger.warning("Shutdown requested, stopping workflow processing...")
                    break

                result = process_workflow(workflow)
                workflow_id = result['workflow_id']

                results[workflow_id] = result['num_chains']
                total_chains += result['num_chains']
                total_steps += result['num_steps']

                # Record detailed stats
                workflow_stats[workflow_id] = {
                    'num_chains': result['num_chains'],
                    'num_steps_in_pattern': result['num_steps_in_pattern'],
                    'pattern': result['pattern'],
                    'duration_seconds': round(result['duration'], 2),
                    'status': 'success' if result['num_chains'] > 0 else 'failed'
                }

                # Increment workflow counter and check if we should save
                self.workflows_since_last_save += 1
                self._save_cache_if_needed(force=False)

        # Final cache save
        logger.info("\n💾 Saving final cache...")
        self._save_cache_if_needed(force=True)

        # Get final cache size
        final_cache_size = sum(len(entries) for entries in self.booking_api._cache.values())
        cache_growth = final_cache_size - initial_cache_size

        # Calculate cache statistics
        total_cache_hits = sum(self.expansion_stats['cache_hits'].values())
        total_attempts = sum(self.expansion_stats['attempts'].values()) if hasattr(self, 'expansion_stats') else 0
        cache_hit_rate = (total_cache_hits / total_attempts * 100) if total_attempts > 0 else 0

        # Print final statistics
        logger.info(f"\n{'='*70}")
        logger.info("EXPANSION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total workflow chains: {total_chains}")
        logger.info(f"Cache growth: {initial_cache_size} → {final_cache_size} entries (+{cache_growth})")
        if total_cache_hits > 0:
            logger.info(f"Cache efficiency: {total_cache_hits} hits ({cache_hit_rate:.1f}% hit rate)")
            logger.info(f"  → Saved {total_cache_hits} API calls via cache")

        # API timing statistics
        api_timings = self.expansion_stats['api_timings']
        if api_timings:
            import statistics
            avg_time = statistics.mean(api_timings)
            median_time = statistics.median(api_timings)
            max_time = max(api_timings)
            slow_calls = sum(1 for t in api_timings if t > 2.0)
            logger.info(f"\nAPI call performance:")
            logger.info(f"  Total calls: {len(api_timings)}")
            logger.info(f"  Average: {avg_time:.2f}s | Median: {median_time:.2f}s | Max: {max_time:.2f}s")
            logger.info(f"  Slow calls (>2s): {slow_calls} ({slow_calls/len(api_timings)*100:.1f}%)")
        logger.info(f"\nPer-workflow results:")
        for workflow_id, num_chains in sorted(results.items()):
            status = workflow_stats[workflow_id]['status']
            status_symbol = '✓' if status == 'success' else '✗'
            logger.info(f"  {status_symbol} {workflow_id}: {num_chains} chains")
        logger.info(f"{'='*70}\n")

        # Save detailed summary report
        from datetime import datetime
        summary_dir = "logs/cache_expansion"
        os.makedirs(summary_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{summary_dir}/summary_{timestamp}.json"

        # Separate successful and failed workflows
        successful_workflows = [wf_id for wf_id, stats in workflow_stats.items() if stats['status'] == 'success']
        failed_workflows = [wf_id for wf_id, stats in workflow_stats.items() if stats['status'] == 'failed']

        summary = {
            'timestamp': timestamp,
            'configuration': {
                'batch_size_step0': batch_size_step0,
                'batch_size_dependent': batch_size_dependent,
                'max_branches_per_step': max_branches_per_step,
                'max_workers': max_workers
            },
            'overall_stats': {
                'total_workflows': len(workflows),
                'successful_workflows': len(successful_workflows),
                'failed_workflows': len(failed_workflows),
                'total_chains': total_chains,
                'cache_growth': cache_growth,
                'initial_cache_size': initial_cache_size,
                'final_cache_size': final_cache_size,
                'cache_hit_rate': round(cache_hit_rate, 2)
            },
            'workflow_details': workflow_stats,
            'successful_workflow_ids': successful_workflows,
            'failed_workflow_ids': failed_workflows
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary report saved to: {summary_file}")

        if failed_workflows:
            logger.warning(f"\n{len(failed_workflows)} workflows failed to generate chains:")
            for wf_id in failed_workflows[:10]:  # Show first 10
                logger.warning(f"  - {wf_id}")
            if len(failed_workflows) > 10:
                logger.warning(f"  ... and {len(failed_workflows) - 10} more (see summary file)")
            logger.info(f"\nTo retry failed workflows, run:")
            logger.info(f"  python expand_cache.py --workflows {','.join(failed_workflows[:5])} ...")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Expand BookingAPI cache with diverse parameter combinations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["independent", "workflow"],
        default="workflow",
        help="Expansion mode: 'workflow' (multi-step chains, recommended) or 'independent' (function-by-function)"
    )

    parser.add_argument(
        "--functions",
        type=str,
        required=False,
        help=(
            "For independent mode: Comma-separated list of function names to expand, or 'all'. "
            "Example: 'Search_Hotels,Search_Flights' or 'all'"
        )
    )

    parser.add_argument(
        "--workflows",
        type=str,
        default=None,
        help=(
            "For workflow mode: Comma-separated workflow IDs, or 'all' for all workflows. "
            "Example: 'workflow_0,workflow_1' or 'all'"
        )
    )

    parser.add_argument(
        "--exclude-workflows",
        type=str,
        default=None,
        help=(
            "For workflow mode: Comma-separated workflow IDs to exclude (skip already processed workflows). "
            "Example: 'workflow_0,workflow_2,workflow_5'"
        )
    )

    parser.add_argument(
        "--max-chains-per-workflow",
        type=int,
        default=None,
        help=(
            "For workflow mode: Maximum chains per workflow. Workflows that already have >= this many chains "
            "(from previous runs) will be skipped. Reads from logs/cache_expansion/summary_*.json files."
        )
    )

    parser.add_argument(
        "--samples-per-function",
        type=int,
        default=50,
        help="Target number of new samples per function"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model for parameter generation"
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to booking_api_cache.json (default: <repo>/environment/booking_api_cache.json)"
    )

    parser.add_argument(
        "--tool-schema-path",
        type=str,
        default=None,
        help="Path to booking_api.json tool schemas (default: auto-detect)"
    )

    parser.add_argument(
        "--batch-size-step0",
        type=int,
        default=50,
        help="For workflow mode: Batch size for independent steps (Step 0). Larger = fewer LLM calls but more API calls."
    )

    parser.add_argument(
        "--batch-size-dependent",
        type=int,
        default=10,
        help="For workflow mode: Batch size for dependent steps (per branch). Larger = fewer LLM calls but more API calls per branch."
    )

    parser.add_argument(
        "--max-branches-per-step",
        type=int,
        default=60,
        help="For workflow mode: Maximum number of candidates to keep per step. Prevents exponential tree growth. Should be >= batch-size-step0."
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="For workflow mode: Number of parallel workers for API calls within each workflow. (default: 10)"
    )

    parser.add_argument(
        "--workflow-workers",
        type=int,
        default=3,
        help="For workflow mode: Number of workflows to process in parallel. Higher = faster overall execution. (default: 3)"
    )

    parser.add_argument(
        "--workflow-templates",
        type=str,
        default="default",
        help="Path to workflow_templates.json (default: data/workflow_templates.json)"
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

    # Console logging
    if args.verbose:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    else:
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    # File logging with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/cache_expansion"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/expansion_{timestamp}.log"
    logger.add(log_file, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.info(f"Logging to file: {log_file}")

    # Check for RAPID_API_KEY
    if "RAPID_API_KEY" not in os.environ:
        logger.error("ERROR: RAPID_API_KEY not found in environment")
        logger.error("Set it with: export RAPID_API_KEY='your_key_here'")
        logger.error("Get a key at: https://rapidapi.com/DataCrawler/api/booking-com15")
        sys.exit(1)

    # Validate arguments based on mode
    if args.mode == "independent":
        if not args.functions:
            logger.error("ERROR: --functions is required for independent mode")
            sys.exit(1)
    elif args.mode == "workflow":
        if not args.workflows:
            logger.error("ERROR: --workflows is required for workflow mode")
            sys.exit(1)

    # Initialize expander
    workflow_templates_path = args.workflow_templates if args.mode == "workflow" else None
    expander = CacheExpander(
        model=args.model,
        cache_path=args.cache_path,
        tool_schema_path=args.tool_schema_path,
        workflow_templates_path=workflow_templates_path,
        random_seed=args.random_seed
    )

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        if expander._shutdown_requested:
            # Second interrupt - force exit
            logger.warning("\n\n⚠️  Force exit requested...")
            sys.exit(1)
        logger.warning("\n\n⚠️  Interrupt received (Ctrl+C), stopping workers...")
        expander._shutdown_requested = True
        # Note: Cache will be saved when threads finish or on second Ctrl+C

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Execute based on mode - wrapped in try/finally to ensure cache is saved
    try:
        if args.mode == "independent":
            # Independent mode (existing behavior)
            if args.functions.lower() == 'all':
                functions = ['all']
            else:
                functions = [f.strip() for f in args.functions.split(',')]

            logger.info(f"Starting cache expansion (INDEPENDENT mode):")
            logger.info(f"  Functions: {functions if functions != ['all'] else 'ALL'}")
            logger.info(f"  Samples per function: {args.samples_per_function}")
            logger.info(f"  Model: {args.model}")
            logger.info("")

            results = expander.expand_cache(functions, args.samples_per_function)

        elif args.mode == "workflow":
            # Workflow mode (new behavior)
            if args.workflows.lower() == 'all':
                workflow_ids = None  # None means all workflows
            else:
                workflow_ids = [w.strip() for w in args.workflows.split(',')]

            # Parse exclude list
            exclude_workflow_ids = None
            if args.exclude_workflows:
                exclude_workflow_ids = [w.strip() for w in args.exclude_workflows.split(',')]

            logger.info(f"Starting cache expansion (WORKFLOW mode):")
            logger.info(f"  Workflows: {workflow_ids if workflow_ids else 'ALL'}")
            if exclude_workflow_ids:
                logger.info(f"  Excluding: {', '.join(exclude_workflow_ids)} ({len(exclude_workflow_ids)} workflows)")
            if args.max_chains_per_workflow:
                logger.info(f"  Max chains per workflow: {args.max_chains_per_workflow}")
            logger.info(f"  Batch sizes: step0={args.batch_size_step0}, dependent={args.batch_size_dependent}")
            logger.info(f"  Max branches per step: {args.max_branches_per_step}")
            logger.info(f"  Parallel workers: {args.max_workers} (API calls), {args.workflow_workers} (workflows)")
            logger.info(f"  Model: {args.model}")
            logger.info("")

            results = expander.expand_workflow_cache(
                workflow_ids=workflow_ids,
                exclude_workflow_ids=exclude_workflow_ids,
                batch_size_step0=args.batch_size_step0,
                batch_size_dependent=args.batch_size_dependent,
                max_branches_per_step=args.max_branches_per_step,
                max_workers=args.max_workers,
                workflow_workers=args.workflow_workers,
                max_chains_per_workflow=args.max_chains_per_workflow
            )

        logger.info("\n✓ Cache expansion complete!")

    finally:
        # Always save cache on exit (normal or interrupted)
        if expander._shutdown_requested:
            logger.info("\n💾 Saving cache before exit...")
        expander._save_cache_if_needed(force=True)
        if expander._shutdown_requested:
            logger.info("✓ Cache saved. Exiting gracefully.")


if __name__ == "__main__":
    main()
