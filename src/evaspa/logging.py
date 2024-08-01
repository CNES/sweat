#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Logging module
"""

import logging
import typing as t

_T = t.TypeVar("_T")


class Singleton(type, t.Generic[_T]):
    """
    Singleton class
    """

    # _instances: dict[Singleton[_T], _T] = {}  # noqa
    _instances = {}  # type: ignore

    def __call__(cls, *args: t.Any, **kwargs: t.Any) -> _T:
        if cls not in cls._instances.keys():
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LoggerManager(object):
    """
    Class to manager logger through the modules
    """

    __metaclass__ = Singleton

    _loggers: dict[str, logging.Logger] = {}

    _level = logging.INFO

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def get_logger(name=None):
        if not name:
            logging.basicConfig(
                level=LoggerManager._level,
                datefmt="%y-%m-%d %H:%M:%S",
                format="%(asctime)s :: %(levelname)s :: %(message)s",
            )
            return logging.getLogger()
        elif name not in LoggerManager._loggers.keys():
            logging.basicConfig(
                level=LoggerManager._level,
                datefmt="%y-%m-%d %H:%M:%S",
                format="%(asctime)s :: %(levelname)s :: %(message)s",
            )
            LoggerManager._loggers[name] = logging.getLogger(str(name))
        return LoggerManager._loggers[name]

    @staticmethod
    def set_level(level):
        LoggerManager._level = level
        for name in LoggerManager._loggers.keys():
            log = LoggerManager._loggers[name]
            log.setLevel(LoggerManager._level)
