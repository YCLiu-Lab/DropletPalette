"""
liquid solver module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import numpy as np
import cantera as ct
from src.core import Grid, Surface
from src.solution import Liquid, LiquidArray
from .math_utils import ThreePointsScheme, EquationProperty
from .liquid_solver_T_evap import LiquidSolverTEvap
from .liquid_solver_Y_evap import LiquidSolverYEvap


class LiquidSolver:
    """液相联合求解器"""
    
    def __init__(self, grid: Grid, liquid_00: Liquid, surface: Surface, solver_type: str = 'ITCID'):
        """初始化液相求解器
        
        Args:
            grid: 网格对象
            liquid_00: 液滴中心液相状态（左边界）
            surface: 表面对象（右边界）
            solver_type: 求解器类型，可选值为'ITCID'（无限传热传质）、'ITCFD'（无限热有限传质）或'FTCFD'（有限传热传质）
        """
        self.grid = grid
        self.liquid_00 = liquid_00
        self.surface = surface
        self.solver_type = solver_type

        # 创建三点格式和方程属性对象
        self.three_points_scheme = ThreePointsScheme.create(self.grid.liquid_grid.cell_count)
        self.equation_property = EquationProperty.create(self.grid.liquid_grid.cell_count, len(self.grid.liquid_array.liquids[0].composition))

        # 创建迭代数组
        self.liquid_array_iter = LiquidArray(self.grid.liquid_array.liquids[0], self.grid.liquid_grid.cell_count)
        self._relative_velocity = np.zeros(self.grid.liquid_grid.cell_count+1)
        # 更新三点格式参数
        self.three_points_scheme.update_scheme(
            self.grid.liquid_grid.lambda_center, 
            self._relative_velocity
        ) 
        
        # 更新方程属性
        self.equation_property.liquid_update_equation_property(
            self.liquid_array_iter, 
            self.grid, 
            self.liquid_00,
            self.surface, 
            self.three_points_scheme, 
            self.liquid_array_iter.Y
        )

        # 创建温度场和质量分数场求解器
        self.T_solver = LiquidSolverTEvap(
            grid=self.grid,
            surface=self.surface,
            liquid_00=self.liquid_00,
            three_points_scheme=self.three_points_scheme,
            equation_property=self.equation_property,
            liquid_array_iter=self.liquid_array_iter
        )
        
        self.y_solver = LiquidSolverYEvap(
            grid=self.grid,
            surface=self.surface,
            liquid_00=self.liquid_00,
            three_points_scheme=self.three_points_scheme,
            equation_property=self.equation_property,
            liquid_array_iter=self.liquid_array_iter,
        )

    def update_scheme_parameters(self):
        """更新数值格式参数"""
        # 更新三点格式参数
        self.three_points_scheme.update_scheme(
            self.grid.liquid_grid.lambda_center, 
            self._relative_velocity
        ) 
        
        # 更新方程属性
        self.equation_property.liquid_update_equation_property(
            self.liquid_array_iter_last, 
            self.grid, 
            self.liquid_00,
            self.surface, 
            self.three_points_scheme, 
            self.liquid_array_iter_last.Y
        )