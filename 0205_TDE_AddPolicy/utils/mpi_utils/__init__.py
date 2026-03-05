# -*- coding: utf-8 -*-
"""
MPI通信工具包
==============

本包提供MPI多进程通信的统一管理接口，用于深度强化学习机器人导航系统。

主要组件：
    - MPIHandler: MPI通信管理类
    - 角色常量：ROLE_MAIN, ROLE_TARGET_PUBLISHER, ROLE_DYNAMIC_OBSTACLE
    - 标签常量：TAG_*系列

作者：Claude Code
创建时间：2025-12-17
版本：1.0.0
"""

from .mpi_handler import (
    MPIHandler,
    ROLE_MAIN,
    ROLE_TARGET_PUBLISHER,
    ROLE_DYNAMIC_OBSTACLE,
    TAG_ROBOT_STATE,
    TAG_PEDESTRIAN_STATES,
    TAG_DYNAMIC_STATE,
    TAG_CURRICULUM_UPDATE,
    TAG_TARGET_UPDATE,
    TAG_SHUTDOWN
)

__all__ = [
    'MPIHandler',
    'ROLE_MAIN',
    'ROLE_TARGET_PUBLISHER',
    'ROLE_DYNAMIC_OBSTACLE',
    'TAG_ROBOT_STATE',
    'TAG_PEDESTRIAN_STATES',
    'TAG_DYNAMIC_STATE',
    'TAG_CURRICULUM_UPDATE',
    'TAG_TARGET_UPDATE',
    'TAG_SHUTDOWN'
]

__version__ = '1.0.0'
