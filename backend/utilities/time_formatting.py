"""
time_formatting.py

Utility functions for parsing and formatting time frames from natural language input.
"""

import logging
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)

def extract_timeframe_from_text(text: str) -> Optional[Dict[str, str]]:
    """Extract timeframe from text and return timeMin and timeMax in ISO format."""
    try:
        now = datetime.now(dt_timezone.utc)
        # Common time frame patterns
        if 'this week' in text.lower():
            # Set time_min to start of current week (Monday)
            start_of_week = now - timedelta(days=now.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            # Set time_max to end of week (Sunday)
            end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return {
                'timeMin': start_of_week.isoformat(),
                'timeMax': end_of_week.isoformat()
            }
        elif 'next week' in text.lower():
            # Set time_min to start of next week (Monday)
            start_of_week = now - timedelta(days=now.weekday()) + timedelta(days=7)
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            # Set time_max to end of next week (Sunday)
            end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return {
                'timeMin': start_of_week.isoformat(),
                'timeMax': end_of_week.isoformat()
            }
        elif 'today' in text.lower():
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1, microseconds=-1)
            return {
                'timeMin': start_of_day.isoformat(),
                'timeMax': end_of_day.isoformat()
            }
        elif 'tomorrow' in text.lower():
            start_of_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1, microseconds=-1)
            return {
                'timeMin': start_of_day.isoformat(),
                'timeMax': end_of_day.isoformat()
            }
        return None
    except Exception as e:
        logger.error(f"Error extracting time frame: {e}")
        return None 