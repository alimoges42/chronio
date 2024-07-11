#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging config for the Chronio package.

Author: Aaron Limoges
"""

import logging
import sys

def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logger = logging.getLogger('chronio')
    logger.setLevel(level)

    # Create console handler and set level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger
