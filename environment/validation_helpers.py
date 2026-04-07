"""
Validation helpers for BookingAPI.

All validators return None if valid, or {"error": "message"} if invalid.
Error messages are detailed and include field names and actual values for debugging.

Design Decisions (2025-10-21):
- Date validation: Format-only (YYYY-MM-DD), no past/future checks
- No API-specific business rules
- Cache is ground-truth: validation happens after cache lookup
- Detailed error messages for debugging
- Time validation: Pattern-only (HH:MM), no range checks
- Driver age: No validation (API handles location-specific rules)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import re


def validate_non_empty_string(value: Any, field_name: str) -> Optional[Dict]:
    """
    Validate string is not None, empty, or whitespace-only.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_non_empty_string("hotel123", "hotel_id")
        None
        >>> validate_non_empty_string("", "hotel_id")
        {'error': "hotel_id cannot be empty or whitespace-only, got: ''"}
        >>> validate_non_empty_string(None, "hotel_id")
        {'error': 'hotel_id is required'}
    """
    if value is None:
        return {"error": f"{field_name} is required"}

    if not isinstance(value, str):
        return {"error": f"{field_name} must be a string, got: {type(value).__name__}"}

    if not value.strip():
        return {"error": f"{field_name} cannot be empty or whitespace-only, got: '{value}'"}

    return None  # Valid


def validate_coordinate(value: str, coord_type: str) -> Optional[Dict]:
    """
    Validate coordinate is a valid numeric string within range.

    Args:
        value: The coordinate value as string
        coord_type: Either "latitude" or "longitude"

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_coordinate("45.5", "latitude")
        None
        >>> validate_coordinate("95.0", "latitude")
        {'error': "latitude must be between -90 and 90, got: '95.0'"}
        >>> validate_coordinate("abc", "longitude")
        {'error': "longitude must be a valid numeric string, got: 'abc'"}
    """
    if not isinstance(value, str):
        return {"error": f"{coord_type} must be a string, got: {type(value).__name__}"}

    # Try to convert to float
    try:
        float_val = float(value)
    except ValueError:
        return {"error": f"{coord_type} must be a valid numeric string, got: '{value}'"}

    # Check range
    if coord_type == "latitude":
        if not (-90 <= float_val <= 90):
            return {"error": f"latitude must be between -90 and 90, got: '{value}'"}
    elif coord_type == "longitude":
        if not (-180 <= float_val <= 180):
            return {"error": f"longitude must be between -180 and 180, got: '{value}'"}
    else:
        return {"error": f"Invalid coord_type: {coord_type}"}

    return None  # Valid


def validate_date_format(date_str: str, field_name: str) -> Optional[Dict]:
    """
    Validate date string is in YYYY-MM-DD format.
    Does NOT check if date is in past or future (per design decision).

    Args:
        date_str: The date string to validate
        field_name: Name of the field for error messages

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_date_format("2024-10-21", "arrival_date")
        None
        >>> validate_date_format("10/21/2024", "arrival_date")
        {'error': "arrival_date must be in YYYY-MM-DD format, got: '10/21/2024'"}
        >>> validate_date_format("2020-01-01", "arrival_date")  # Historical allowed
        None
    """
    if not isinstance(date_str, str):
        return {"error": f"{field_name} must be a string, got: {type(date_str).__name__}"}

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return {"error": f"{field_name} must be in YYYY-MM-DD format, got: '{date_str}'"}

    return None  # Valid


def validate_date_range(start_date: str, end_date: str,
                       start_name: str = "start_date",
                       end_name: str = "end_date") -> Optional[Dict]:
    """
    Validate end_date is after start_date.
    Assumes both dates are already validated for format.

    Args:
        start_date: The start date string (YYYY-MM-DD)
        end_date: The end date string (YYYY-MM-DD)
        start_name: Name of start date field for error messages
        end_name: Name of end date field for error messages

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_date_range("2024-10-21", "2024-10-25")
        None
        >>> validate_date_range("2024-10-21", "2024-10-21")
        {'error': "end_date must be after start_date, got: start='2024-10-21', end='2024-10-21'"}
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        # This shouldn't happen if dates were validated first
        return {"error": f"Dates must be in YYYY-MM-DD format"}

    if start >= end:
        return {"error": f"{end_name} must be after {start_name}, got: start='{start_date}', end='{end_date}'"}

    return None  # Valid


def validate_enum(value: str, field_name: str, allowed_values: Set[str]) -> Optional[Dict]:
    """
    Validate value is in the set of allowed values.
    Case-sensitive by design decision.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        allowed_values: Set of allowed values

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_enum("ECONOMY", "cabinClass", {"ECONOMY", "BUSINESS", "FIRST"})
        None
        >>> validate_enum("economy", "cabinClass", {"ECONOMY", "BUSINESS", "FIRST"})
        {'error': "cabinClass must be one of: BUSINESS, ECONOMY, FIRST, got: 'economy'"}
    """
    if value not in allowed_values:
        allowed_str = ', '.join(sorted(allowed_values))
        return {"error": f"{field_name} must be one of: {allowed_str}, got: '{value}'"}

    return None  # Valid


def validate_numeric_range(value: float, field_name: str,
                          min_val: float = None,
                          max_val: float = None) -> Optional[Dict]:
    """
    Validate numeric value is within range (if specified).

    Args:
        value: The numeric value to validate
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_numeric_range(2.0, "adults", min_val=1)
        None
        >>> validate_numeric_range(0.5, "adults", min_val=1)
        {'error': 'adults must be at least 1, got: 0.5'}
    """
    if not isinstance(value, (int, float)):
        return {"error": f"{field_name} must be a number, got: {type(value).__name__}"}

    if min_val is not None and value < min_val:
        return {"error": f"{field_name} must be at least {min_val}, got: {value}"}

    if max_val is not None and value > max_val:
        return {"error": f"{field_name} must be at most {max_val}, got: {value}"}

    return None  # Valid


def validate_list_structure(items: List, field_name: str,
                           required_fields: List[str]) -> Optional[Dict]:
    """
    Validate list of dicts has required fields.
    Also validates each required field is non-empty if it's a string.

    Args:
        items: The list to validate
        field_name: Name of the field for error messages
        required_fields: List of required field names in each dict

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_list_structure([{"fromId": "SFO", "toId": "LAX"}], "legs", ["fromId", "toId"])
        None
        >>> validate_list_structure([], "legs", ["fromId"])
        {'error': 'legs cannot be empty'}
        >>> validate_list_structure([{"fromId": "SFO"}], "legs", ["fromId", "toId"])
        {'error': 'legs[0] missing required field: toId'}
    """
    if not isinstance(items, list):
        return {"error": f"{field_name} must be a list, got: {type(items).__name__}"}

    if not items:
        return {"error": f"{field_name} cannot be empty"}

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return {"error": f"{field_name}[{i}] must be a dictionary, got: {type(item).__name__}"}

        for field in required_fields:
            if field not in item:
                return {"error": f"{field_name}[{i}] missing required field: {field}"}

            # Validate field is non-empty if it's a string
            value = item[field]
            if isinstance(value, str) and not value.strip():
                return {"error": f"{field_name}[{i}].{field} cannot be empty, got: '{value}'"}

    return None  # Valid


def validate_time_format(time_str: str, field_name: str) -> Optional[Dict]:
    """
    Validate time format HH:MM (pattern only, no range checking).
    Per design decision: format-only, don't validate hour/minute ranges.

    Args:
        time_str: The time string to validate
        field_name: Name of the field for error messages

    Returns:
        None if valid, {"error": "message"} if invalid

    Examples:
        >>> validate_time_format("14:30", "pick_up_time")
        None
        >>> validate_time_format("9:30", "pick_up_time")
        {'error': "pick_up_time must be in HH:MM format, got: '9:30'"}
        >>> validate_time_format("25:99", "pick_up_time")  # Invalid but pattern matches
        None  # We don't validate ranges per design decision
    """
    if not isinstance(time_str, str):
        return {"error": f"{field_name} must be a string, got: {type(time_str).__name__}"}

    if not re.match(r'^\d{2}:\d{2}$', time_str):
        return {"error": f"{field_name} must be in HH:MM format, got: '{time_str}'"}

    return None  # Valid
