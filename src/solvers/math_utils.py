"""
math utils module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

from dataclasses import dataclass
import numpy as np
import cantera as ct
from numba import jit
import warnings
from src.core import Grid, Surface
from src.solution import Liquid, LiquidArray

# 三点格式类，管理各种数值离散化方法, 当速度发生改变时，需要重新计算五点格式系数
# 参数计算类，计算数值计算过程中出现的参数，当气液相状态发生改变时，需要重新计算参数计算类，其中又包括计算能量方程的参数，和计算组分守恒方程的参数
@dataclass(eq=False, order=False, unsafe_hash=False)
class ThreePointsScheme:
    """三点格式类，管理各种数值离散化方法
    
    Attributes:
        cell_count (int): 网格单元数量
        scheme_center_left (np.ndarray): 左边界中心差分系数 [cell_count, 3]
        scheme_center_right (np.ndarray): 右边界中心差分系数 [cell_count, 3]
        scheme_upwind_left (np.ndarray): 左边界一阶迎风系数 [cell_count, 3]
        scheme_upwind_right (np.ndarray): 右边界一阶迎风系数 [cell_count, 3]
    """
    __slots__ = ('cell_count', 'scheme_center_left', 'scheme_center_right',
                'scheme_upwind_left', 'scheme_upwind_right')
    
    cell_count: int
    scheme_center_left: np.ndarray
    scheme_center_right: np.ndarray
    scheme_upwind_left: np.ndarray
    scheme_upwind_right: np.ndarray

    
    @classmethod
    def create(cls, cell_count: int) -> 'ThreePointsScheme':
        """创建三点格式数据结构"""
        return cls(
            cell_count=cell_count,
            scheme_center_left=np.zeros((cell_count, 3)),
            scheme_center_right=np.zeros((cell_count, 3)),
            scheme_upwind_left=np.zeros((cell_count, 3)),
            scheme_upwind_right=np.zeros((cell_count, 3))
        )
    
    def update_scheme(self, lambda_center: np.ndarray, relative_velocity_all: np.ndarray):
        """更新数值格式系数
        
        Args:
            lambda_center: 中心差分格式系数
            relative_velocity_all: 相对速度(包含左边界处的速度)
        """
        # 初始化数组
        self.scheme_center_left.fill(0)
        self.scheme_center_right.fill(0)
        self.scheme_upwind_left.fill(0)
        self.scheme_upwind_right.fill(0)
        
        # 左边界：
        self.scheme_center_left[0, :] = [1/2, 1/2, 0]  # 第一个网格使用一阶格式
        self.scheme_center_left[1:, 0] = 1 - lambda_center[:-1]  # 中间项
        self.scheme_center_left[1:, 1] = lambda_center[:-1]  # 右侧项
        
        # 右边界：
        # - 最后一个网格：[0, 1, 0] (一阶格式)
        # - 其他网格应该是 [0, 1-λ, λ]
        self.scheme_center_right[:-1, 1] = 1 - lambda_center[:-1] # 左侧项
        self.scheme_center_right[:-1, 2] = lambda_center[:-1]  # 中间项
        self.scheme_center_right[-1, :] = [0, 1/2, 1/2]  # 最后一个网格使用一阶格式
        
        # 更新一阶迎风格式系数
        # 左边界迎风格式：
        # - 速度 >= 0: [1, 0, 0] (使用左侧  点)
        # - 速度 < 0: [0, 1, 0] (使用当前点)
        for i in range(self.cell_count):
            if relative_velocity_all[i] >= 0:
                self.scheme_upwind_left[i, 0] = 1  # 使用左侧点
            else:
                self.scheme_upwind_left[i, 1] = 1  # 使用当前点
        
        # 右边界迎风格式：
        # - 速度 >= 0: [0, 1, 0] (使用当前点)
        # - 速度 < 0: [0, 0, 1] (使用右侧点)
        for i in range(self.cell_count):
            if relative_velocity_all[i+1] >= 0:
                self.scheme_upwind_right[i, 1] = 1  # 使用当前点
            else:
                self.scheme_upwind_right[i, 2] = 1  # 使用右侧点


@dataclass(eq=False, order=False, unsafe_hash=False)
class EquationProperty:
    """参数计算类，计算数值计算过程中出现的参数"""
    __slots__ = ('specific_enthalpy_left_upwind', 'specific_enthalpy_right_upwind',          
                 'density_left_upwind', 'density_right_upwind',
                 'thermal_conductivity_left_center', 'thermal_conductivity_right_center',
                 'rho_diffusivity_left_center', 'rho_diffusivity_right_center',
                 'temperature_left_upwind', 'temperature_right_upwind',
                 'mass_fraction_left_upwind', 'mass_fraction_right_upwind',
                 'temperature_gradient_left', 'temperature_gradient_right',
                 'mass_fraction_gradient_left', 'mass_fraction_gradient_right',
                 'mass_flux_diffusion', 'mass_flux_convection',
                 'energy_flux_diffusion', 'energy_flux_convection')
    
    specific_enthalpy_left_upwind: np.ndarray
    specific_enthalpy_right_upwind: np.ndarray
    density_left_upwind: np.ndarray
    density_right_upwind: np.ndarray

    thermal_conductivity_left_center: np.ndarray
    thermal_conductivity_right_center: np.ndarray
    rho_diffusivity_left_center: np.ndarray
    rho_diffusivity_right_center: np.ndarray
    
    temperature_left_upwind: np.ndarray
    temperature_right_upwind: np.ndarray
    mass_fraction_left_upwind: np.ndarray
    mass_fraction_right_upwind: np.ndarray

    temperature_gradient_left: np.ndarray
    temperature_gradient_right: np.ndarray
    mass_fraction_gradient_left: np.ndarray
    mass_fraction_gradient_right: np.ndarray
    
    mass_flux_diffusion: np.ndarray
    mass_flux_convection: np.ndarray
    energy_flux_diffusion: np.ndarray
    energy_flux_convection: np.ndarray

    @classmethod
    def create(cls, cell_count: int, n_species: int) -> 'EquationProperty':
        return cls(
            specific_enthalpy_left_upwind=np.zeros(cell_count),
            specific_enthalpy_right_upwind=np.zeros(cell_count),
            density_left_upwind=np.zeros(cell_count),
            density_right_upwind=np.zeros(cell_count),
            thermal_conductivity_left_center=np.zeros(cell_count),
            thermal_conductivity_right_center=np.zeros(cell_count),
            rho_diffusivity_left_center=np.zeros((cell_count,n_species)),
            rho_diffusivity_right_center=np.zeros((cell_count,n_species)),
            temperature_left_upwind=np.zeros(cell_count),
            temperature_right_upwind=np.zeros(cell_count),
            mass_fraction_left_upwind=np.zeros((cell_count,n_species)),
            mass_fraction_right_upwind=np.zeros((cell_count,n_species)),
            temperature_gradient_left=np.zeros(cell_count),
            temperature_gradient_right=np.zeros(cell_count),
            mass_fraction_gradient_left=np.zeros((cell_count,n_species)),
            mass_fraction_gradient_right=np.zeros((cell_count,n_species)),
            mass_flux_diffusion = np.zeros((cell_count,n_species)),
            mass_flux_convection = np.zeros((cell_count,n_species)),
            energy_flux_diffusion = np.zeros(cell_count),
            energy_flux_convection = np.zeros(cell_count)
        )
    


    def check_all_arrays(self):
        """检查所有数组中的负值并发出警告"""
        arrays_to_check = {
            "specific_enthalpy_left_upwind": self.specific_enthalpy_left_upwind,
            "specific_enthalpy_right_upwind": self.specific_enthalpy_right_upwind,
            "density_left_upwind": self.density_left_upwind,
            "density_right_upwind": self.density_right_upwind,
            "thermal_conductivity_left_center": self.thermal_conductivity_left_center,
            "thermal_conductivity_right_center": self.thermal_conductivity_right_center,
            "rho_diffusivity_left_center": self.rho_diffusivity_left_center,
            "rho_diffusivity_right_center": self.rho_diffusivity_right_center,
            "temperature_left_upwind": self.temperature_left_upwind,
            "temperature_right_upwind": self.temperature_right_upwind,
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }
        
        for name, array in arrays_to_check.items():
            if np.any(array < 0):
                # 找到所有负值的位置
                negative_indices = np.where(array < 0)
                if len(array.shape) == 1:
                    indices_str = f"位置: {negative_indices[0]}"
                else:
                    indices_str = f"位置: {list(zip(*negative_indices))}"
                warnings.warn(f"在 {name} 中检测到负值，最小值: {np.min(array)}，{indices_str}")

    def check_rho_diffusivity_zeros(self):
        """检查扩散系数中的0值"""
        if np.any(self.rho_diffusivity_left_center == 0) or np.any(self.rho_diffusivity_right_center == 0):
            warnings.warn("在扩散系数中检测到0值")

    def gas_update_equation_property_T(self, gas_array_iter: ct.SolutionArray, grid: Grid, surface: 'Surface', gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme):
        """更新气相方程中与温度相关的参数
        
        Args:
            gas_array_iter: 气相状态数组
            grid: 网格对象
            surface: 表面对象
            gas_inf: 无穷远气相状态
            three_points_scheme: 三点格式系数
        """
        # 更新比焓
        self.specific_enthalpy_left_upwind = calculate_three_point_array(
            gas_array_iter.cp_mass * gas_array_iter.density_mass,
            surface.gas_surface.cp_mass * surface.gas_surface.density_mass,
            gas_inf.cp_mass * gas_inf.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count
        )
        self.specific_enthalpy_right_upwind = calculate_three_point_array(
            gas_array_iter.cp_mass * gas_array_iter.density_mass,
            surface.gas_surface.cp_mass * surface.gas_surface.density_mass,
            gas_inf.cp_mass * gas_inf.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count
        )

        # 更新热导率
        self.thermal_conductivity_left_center = calculate_three_point_array(
            gas_array_iter.thermal_conductivity,
            surface.gas_surface.thermal_conductivity,
            gas_inf.thermal_conductivity,
            three_points_scheme.scheme_center_left,
            grid.gas_grid.cell_count
        )
        self.thermal_conductivity_right_center = calculate_three_point_array(
            gas_array_iter.thermal_conductivity,
            surface.gas_surface.thermal_conductivity,
            gas_inf.thermal_conductivity,
            three_points_scheme.scheme_center_right,
            grid.gas_grid.cell_count
        )

        # 更新温度
        self.temperature_left_upwind = calculate_three_point_array(
            gas_array_iter.T,
            surface.gas_surface.T,
            gas_inf.T,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count
        )
        self.temperature_right_upwind = calculate_three_point_array(
            gas_array_iter.T,
            surface.gas_surface.T,
            gas_inf.T,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count
        )

        # 计算温度梯度
        self.temperature_gradient_left, self.temperature_gradient_right = calculate_temperature_gradients(
            gas_array_iter.T, grid.gas_grid.positions_volume_centers,
            surface.gas_surface.T, gas_inf.T,
            grid.params.droplet_radius, grid.params.r_inf
        )

        # 检查温度相关数组中的负值
        for name, array in {
            "specific_enthalpy_left_upwind": self.specific_enthalpy_left_upwind,
            "specific_enthalpy_right_upwind": self.specific_enthalpy_right_upwind,
            "thermal_conductivity_left_center": self.thermal_conductivity_left_center,
            "thermal_conductivity_right_center": self.thermal_conductivity_right_center,
            "temperature_left_upwind": self.temperature_left_upwind,
            "temperature_right_upwind": self.temperature_right_upwind
        }.items():
            if np.any(array < 0):
                negative_indices = np.where(array < 0)
                indices_str = f"位置: {negative_indices[0]}"
                warnings.warn(f"在 {name} 中检测到负值，最小值: {np.min(array)}，{indices_str}")
        
   
    def gas_update_equation_property_Y_fast(self, gas_array_iter: ct.SolutionArray, grid: Grid, surface: 'Surface', gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """更新气相方程中与组分质量分数相关的参数
        
        Args:
            gas_array_iter: 气相状态数组
            grid: 网格对象
            surface: 表面对象
        """
                # 更新质量分数
        self.mass_fraction_left_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )
        self.mass_fraction_right_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )

        # 计算组分质量分数梯度
        self.mass_fraction_gradient_left, self.mass_fraction_gradient_right = calculate_species_gradients(
            mass_fraction, grid.gas_grid.positions_volume_centers,
            surface.gas_surface.Y, gas_inf.Y,
            grid.params.droplet_radius, grid.params.r_inf
        )
                # 检查组分相关数组中的负值
        for name, array in {
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }.items():
            if np.any(array < -1e-3):
                negative_indices = np.where(array < -1e-3)
                if len(array.shape) == 1:
                    indices_str = f"位置: {negative_indices[0]}"
                else:
                    indices_str = f"位置: {list(zip(*negative_indices))}"
                warnings.warn(f"在 {name} 中检测到负值，最小值: {np.min(array)}，{indices_str}")

    def gas_update_equation_property_Y(self, gas_array_iter: ct.SolutionArray, grid: Grid, surface: 'Surface', gas_inf: ct.Solution, three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """更新气相方程中与组分质量分数相关的参数
        
        Args:
            gas_array_iter: 气相状态数组
            grid: 网格对象
            surface: 表面对象
            gas_inf: 无穷远气相状态
            three_points_scheme: 三点格式系数
            mass_fraction: 质量分数数组
        """
        # 计算密度和扩散系数的乘积
        rho_diffusivity = gas_array_iter.density_mass[:, np.newaxis] * gas_array_iter.mix_diff_coeffs_mass
        rho_diffusivity_inf = gas_inf.density_mass * gas_inf.mix_diff_coeffs_mass
        rho_diffusivity_surface = surface.gas_surface.density_mass * surface.gas_surface.mix_diff_coeffs_mass
        # 更新密度
        self.density_left_upwind = calculate_three_point_array(
            gas_array_iter.density_mass,
            surface.gas_surface.density_mass,
            gas_inf.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count
        )
        self.density_right_upwind = calculate_three_point_array(
            gas_array_iter.density_mass,
            surface.gas_surface.density_mass,
            gas_inf.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count
        )

        # 更新扩散系数
        self.rho_diffusivity_left_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_surface,
            rho_diffusivity_inf,
            three_points_scheme.scheme_center_left,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )
        self.rho_diffusivity_right_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_surface,
            rho_diffusivity_inf,
            three_points_scheme.scheme_center_right,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )

        # 更新质量分数
        self.mass_fraction_left_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_left,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )
        self.mass_fraction_right_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            surface.gas_surface.Y,
            gas_inf.Y,
            three_points_scheme.scheme_upwind_right,
            grid.gas_grid.cell_count,
            gas_array_iter.n_species
        )

        # 计算组分质量分数梯度
        self.mass_fraction_gradient_left, self.mass_fraction_gradient_right = calculate_species_gradients(
            mass_fraction, grid.gas_grid.positions_volume_centers,
            surface.gas_surface.Y, gas_inf.Y,
            grid.params.droplet_radius, grid.params.r_inf
        )

        # 检查组分相关数组中的负值
        for name, array in {
            "density_left_upwind": self.density_left_upwind,
            "density_right_upwind": self.density_right_upwind,
            "rho_diffusivity_left_center": self.rho_diffusivity_left_center,
            "rho_diffusivity_right_center": self.rho_diffusivity_right_center,
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }.items():
            if np.any(array < -1e-3):
                negative_indices = np.where(array < -1e-3)
                if len(array.shape) == 1:
                    indices_str = f"位置: {negative_indices[0]}"
                else:
                    indices_str = f"位置: {list(zip(*negative_indices))}"
                warnings.warn(f"在 {name} 中检测到负值，最小值: {np.min(array)}，{indices_str}")
        
        # 检查扩散系数中的0值
        self.check_rho_diffusivity_zeros()

    def liquid_update_equation_property(self, liquid_array_iter: 'LiquidArray', grid: Grid, liquid_00: 'Liquid', surface: 'Surface', three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """更新液相方程参数
        
        Args:
            liquid_array_iter: 液相状态数组
            grid: 网格对象
            liquid_00: 液滴中心液相状态（左边界）
            surface: 表面对象（右边界）
            three_points_scheme: 三点格式系数
            mass_fraction: 质量分数数组
        """
        # 更新温度相关的参数
        self.liquid_update_equation_property_T(liquid_array_iter, grid, liquid_00, surface, three_points_scheme)
        
        # 更新组分相关的参数
        self.liquid_update_equation_property_Y(liquid_array_iter, grid, liquid_00, surface, three_points_scheme, mass_fraction)

    def liquid_update_equation_property_T(self, liquid_array_iter: 'LiquidArray', grid: Grid, liquid_00: 'Liquid', surface: 'Surface', three_points_scheme: ThreePointsScheme):
        """更新液相方程中与温度相关的参数
        
        Args:
            liquid_array_iter: 液相状态数组
            grid: 网格对象
            liquid_00: 液滴中心液相状态（左边界）
            surface: 表面对象（右边界）
            three_points_scheme: 三点格式系数
        """
        # 更新比焓
        self.specific_enthalpy_left_upwind = calculate_three_point_array(
            liquid_array_iter.cp_mass * liquid_array_iter.density_mass,
            liquid_00.cp_mass * liquid_00.density_mass,
            surface.liquid_surface.cp_mass * surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count
        )
        self.specific_enthalpy_right_upwind = calculate_three_point_array(
            liquid_array_iter.cp_mass * liquid_array_iter.density_mass,
            liquid_00.cp_mass * liquid_00.density_mass,
            surface.liquid_surface.cp_mass * surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count
        )

        # 更新热导率
        self.thermal_conductivity_left_center = calculate_three_point_array(
            liquid_array_iter.thermal_conductivity,
            liquid_00.thermal_conductivity,
            surface.liquid_surface.thermal_conductivity,
            three_points_scheme.scheme_center_left,
            grid.liquid_grid.cell_count
        )
        self.thermal_conductivity_right_center = calculate_three_point_array(
            liquid_array_iter.thermal_conductivity,
            liquid_00.thermal_conductivity,
            surface.liquid_surface.thermal_conductivity,
            three_points_scheme.scheme_center_right,
            grid.liquid_grid.cell_count
        )

        # 更新温度
        self.temperature_left_upwind = calculate_three_point_array(
            liquid_array_iter.T,
            liquid_00.temperature,
            surface.liquid_surface.temperature,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count
        )
        self.temperature_right_upwind = calculate_three_point_array(
            liquid_array_iter.T,
            liquid_00.temperature,
            surface.liquid_surface.temperature,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count
        )

        # 计算温度梯度
        self.temperature_gradient_left, self.temperature_gradient_right = calculate_temperature_gradients(
            liquid_array_iter.T, grid.liquid_grid.positions_volume_centers,
            liquid_00.temperature, surface.liquid_surface.temperature,
            0.0, grid.params.droplet_radius  # 液相的左边界是0，右边界是液滴半径
        )

        # 检查温度相关数组中的负值
        for name, array in {
            "specific_enthalpy_left_upwind": self.specific_enthalpy_left_upwind,
            "specific_enthalpy_right_upwind": self.specific_enthalpy_right_upwind,
            "thermal_conductivity_left_center": self.thermal_conductivity_left_center,
            "thermal_conductivity_right_center": self.thermal_conductivity_right_center,
            "temperature_left_upwind": self.temperature_left_upwind,
            "temperature_right_upwind": self.temperature_right_upwind
        }.items():
            if np.any(array < 0):
                negative_indices = np.where(array < 0)
                indices_str = f"位置: {negative_indices[0]}"
                warnings.warn(f"在 {name} 中检测到负值，最小值: {np.min(array)}，{indices_str}")

    def liquid_update_equation_property_Y(self, liquid_array_iter: 'LiquidArray', grid: Grid, liquid_00: 'Liquid', surface: 'Surface', three_points_scheme: ThreePointsScheme, mass_fraction: np.ndarray):
        """更新液相方程中与组分质量分数相关的参数
        
        Args:
            liquid_array_iter: 液相状态数组
            grid: 网格对象
            liquid_00: 液滴中心液相状态（左边界）
            surface: 表面对象（右边界）
            three_points_scheme: 三点格式系数
            mass_fraction: 质量分数数组
        """
        # 计算密度和扩散系数的乘积
        rho_diffusivity = liquid_array_iter.density_mass[:, np.newaxis] * liquid_array_iter.diffusion_mean
        rho_diffusivity_00 = liquid_00.density_mass * liquid_00.diffusion_mean
        rho_diffusivity_surface = surface.liquid_surface.density_mass * surface.liquid_surface.diffusion_mean

        # 更新密度
        self.density_left_upwind = calculate_three_point_array(
            liquid_array_iter.density_mass,
            liquid_00.density_mass,
            surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count
        )
        self.density_right_upwind = calculate_three_point_array(
            liquid_array_iter.density_mass,
            liquid_00.density_mass,
            surface.liquid_surface.density_mass,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count
        )

        # 更新扩散系数
        self.rho_diffusivity_left_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_00,
            rho_diffusivity_surface,
            three_points_scheme.scheme_center_left,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )
        self.rho_diffusivity_right_center = calculate_multispecies_three_point_array(
            rho_diffusivity,
            rho_diffusivity_00,
            rho_diffusivity_surface,
            three_points_scheme.scheme_center_right,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )

        # 更新质量分数
        self.mass_fraction_left_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            liquid_00.mass_fraction,
            surface.liquid_surface.mass_fraction,
            three_points_scheme.scheme_upwind_left,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )
        self.mass_fraction_right_upwind = calculate_multispecies_three_point_array(
            mass_fraction,
            liquid_00.mass_fraction,
            surface.liquid_surface.mass_fraction,
            three_points_scheme.scheme_upwind_right,
            grid.liquid_grid.cell_count,
            len(liquid_array_iter.liquids[0].composition)
        )

        # 计算组分质量分数梯度
        self.mass_fraction_gradient_left, self.mass_fraction_gradient_right = calculate_species_gradients(
            mass_fraction, grid.liquid_grid.positions_volume_centers,
            liquid_00.mass_fraction, surface.liquid_surface.mass_fraction,
            0.0, grid.params.droplet_radius  # 液相的左边界是0，右边界是液滴半径
        )

        # 检查组分相关数组中的负值
        for name, array in {
            "density_left_upwind": self.density_left_upwind,
            "density_right_upwind": self.density_right_upwind,
            "rho_diffusivity_left_center": self.rho_diffusivity_left_center,
            "rho_diffusivity_right_center": self.rho_diffusivity_right_center,
            "mass_fraction_left_upwind": self.mass_fraction_left_upwind,
            "mass_fraction_right_upwind": self.mass_fraction_right_upwind
        }.items():
            if np.any(array < 0):
                negative_indices = np.where(array < 0)
                if len(array.shape) == 1:
                    indices_str = f"位置: {negative_indices[0]}"
                else:
                    indices_str = f"位置: {list(zip(*negative_indices))}"
                warnings.warn(f"在 {name} 中检测到负值，最小值: {np.min(array)}，{indices_str}")
        
        # 检查扩散系数中的0值
        self.check_rho_diffusivity_zeros()



@jit(nopython=True, cache=True)
def check_arrays_min(arrays):
    return np.min(arrays)



@jit(nopython=True, cache=True)
def calculate_three_point_array(mid_values: np.ndarray, left_boundary_value: float, 
                          right_boundary_value: float,scheme_array: np.ndarray, n_cells: int) -> np.ndarray:
    """计算三点数组"""
    three_point_array = create_three_point_array(mid_values, left_boundary_value, right_boundary_value, n_cells)
    return np.sum(three_point_array * scheme_array, axis=1)


@jit(nopython=True, cache=True)
def create_three_point_array(mid_values: np.ndarray, left_boundary_value: float, 
                          right_boundary_value: float, n_cells: int) -> np.ndarray:
    """创建三点数组 [左, 中, 右]
    
    Args:
        mid_values: 中间网格的值（当前网格）
        left_boundary_value: 左边界值
        right_boundary_value: 右边界值
        n_cells: 网格数量
        
    Returns:
        np.ndarray: 形状为(n_cells, 3)的三点数组
    """
    result = np.zeros((n_cells, 3))
    
    # 中间值（索引1）
    result[:, 1] = mid_values
    
    # 左侧值（索引0）
    result[0, 0] = left_boundary_value  # 第一个网格使用左边界值
    result[1:, 0] = mid_values[:-1]     # 其他网格使用前一个网格的值
    
    # 右侧值（索引2）
    result[:-1, 2] = mid_values[1:]     # 非最后一个网格使用后一个网格的值
    result[-1, 2] = right_boundary_value # 最后一个网格使用右边界值
    
    return result

@jit(nopython=True, cache=True)
def create_multispecies_three_point_array(mid_values: np.ndarray, left_boundary_value: np.ndarray,
                                       right_boundary_value: np.ndarray, n_cells: int, n_species: int) -> np.ndarray:
    """为多组分物性参数创建三点数组
    
    Args:
        mid_values: 中间网格的值，形状为(n_cells, n_species)
        left_boundary_value: 左边界值，形状为(n_species,)
        right_boundary_value: 右边界值，形状为(n_species,)
        n_cells: 网格数量
        n_species: 组分数量
        
    Returns:
        np.ndarray: 形状为(n_cells, 3, n_species)的三点数组
    """
    # 预分配结果数组
    result = np.empty((n_cells, 3, n_species))
    
    # 中间值（索引1）
    result[:, 1, :] = mid_values
    
    # 左侧值（索引0）
    result[1:, 0, :] = mid_values[:-1, :]
    result[0, 0, :] = left_boundary_value
    
    # 右侧值（索引2）
    result[:-1, 2, :] = mid_values[1:, :]
    result[-1, 2, :] = right_boundary_value
    
    return result

@jit(nopython=True, cache=True)
def calculate_multispecies_three_point_array(mid_values: np.ndarray, left_boundary_value: np.ndarray,
                                         right_boundary_value: np.ndarray, scheme_array: np.ndarray,
                                         n_cells: int, n_species: int) -> np.ndarray:
    """计算多组分物性参数的三点数组
    
    Args:
        mid_values: 中间网格的值，形状为(n_cells, n_species)
        left_boundary_value: 左边界值，形状为(n_species,)
        right_boundary_value: 右边界值，形状为(n_species,)
        scheme_array: 数值格式系数数组，形状为(n_cells, 3)
        n_cells: 网格数量
        n_species: 组分数量
        
    Returns:
        np.ndarray: 形状为(n_cells, n_species)的计算结果
    """
    # 创建三点数组
    three_point_array = create_multispecies_three_point_array(
        mid_values, left_boundary_value, right_boundary_value, n_cells, n_species
    )
    
    # 初始化结果数组
    result = np.empty((n_cells, n_species), dtype=np.float64)
    
    # 并行计算
    for i in range(n_cells):  
        for k in range(n_species):
            result[i, k] = np.sum(three_point_array[i, :, k] * scheme_array[i, :])
    
    return result

@jit(nopython=True, cache=True)
def calculate_temperature_gradients(temperature: np.ndarray, volume_centers: np.ndarray,
                                  surface_temp: float, gas_inf_temp: float,
                                  droplet_radius: float, r_inf: float) -> tuple[np.ndarray, np.ndarray]:
    """计算温度梯度
    
    Args:
        temperature: 温度数组
        volume_centers: 体积中心坐标数组
        surface_temp: 表面温度
        gas_inf_temp: 无穷远处温度
        droplet_radius: 液滴半径
        r_inf: 无穷远处半径
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (左梯度, 右梯度)
    """
    n = len(temperature)
    gradient_left = np.zeros(n)
    gradient_right = np.zeros(n)
    
    # 计算内部网格的温度梯度
    gradient_left[1:] = (temperature[1:] - temperature[:-1]) / (volume_centers[1:] - volume_centers[:-1])
    gradient_right[:-1] = gradient_left[1:]
    
    # 处理边界网格的温度梯度
    gradient_left[0] = (temperature[0] - surface_temp) / (volume_centers[0] - droplet_radius)
    gradient_right[-1] = (gas_inf_temp - temperature[-1]) / (r_inf - volume_centers[-1])
    
    return gradient_left, gradient_right

@jit(nopython=True, cache=True)
def calculate_species_gradients(mass_fraction: np.ndarray, volume_centers: np.ndarray,
                              surface_mass_fraction: np.ndarray, gas_inf_mass_fraction: np.ndarray,
                              droplet_radius: float, r_inf: float) -> tuple[np.ndarray, np.ndarray]:
    """计算组分质量分数梯度
    
    Args:
        mass_fraction: 质量分数数组，形状为(n_cells, n_species)
        volume_centers: 体积中心坐标数组
        surface_mass_fraction: 表面质量分数数组，形状为(n_species,)
        gas_inf_mass_fraction: 无穷远处质量分数数组，形状为(n_species,)
        droplet_radius: 液滴半径
        r_inf: 无穷远处半径
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (左梯度, 右梯度)，形状均为(n_cells, n_species)
    """
    n_cells, n_species = mass_fraction.shape
    gradient_left = np.zeros((n_cells, n_species))
    gradient_right = np.zeros((n_cells, n_species))
    
    # 计算内部网格的质量分数梯度
    for i in range(1, n_cells):
        for j in range(n_species):
            gradient_left[i, j] = (mass_fraction[i, j] - mass_fraction[i-1, j]) / (volume_centers[i] - volume_centers[i-1])
            gradient_right[i-1, j] = gradient_left[i, j]
    
    # 处理边界网格的质量分数梯度
    for j in range(n_species):
        gradient_left[0, j] = (mass_fraction[0, j] - surface_mass_fraction[j]) / (volume_centers[0] - droplet_radius)
        gradient_right[-1, j] = (gas_inf_mass_fraction[j] - mass_fraction[-1, j]) / (r_inf - volume_centers[-1])
    
    return gradient_left, gradient_right