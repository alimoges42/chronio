#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exceptions for the Chronio package.

Author: Aaron Limoges
"""

class ChronIOError(Exception):
    """Base exception class for ChronIO."""
    pass

class InputError(ChronIOError):
    """Exception raised for errors in the input."""
    pass

class ProcessingError(ChronIOError):
    """Exception raised when an error occurs during data processing."""
    pass

class IOError(ChronIOError):
    """Exception raised for input/output related errors."""
    pass
