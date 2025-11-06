"""
surface module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.


This module contains two main classes:
1. SurfaceState: surface state data class
2. Surface: surface management class

main functions:
- surface state management
- gas-liquid interface mass and heat transfer
- component balance calculation
"""

import numpy as np
import cantera as ct
from scipy import stats
from scipy.special import gammainc
from importlib.resources import files
from dataclasses import dataclass, field
from .grid import Grid
from src.solution import init_species_mapping, Liquid
from src.solution import liquid_para

@dataclass
class SurfaceState:
    """surface state data class
    
    used to store and manage the state parameters of the gas-liquid interface
    
    Attributes:
        temperature: temperature [K]
        stefan_velocity_liquid: liquid phase Stefan velocity [m/s]
        stefan_velocity_gas: gas phase Stefan velocity [m/s]
        old_stefan_velocity_liquid: previous time liquid phase Stefan velocity [m/s]
        old_stefan_velocity_gas: previous time gas phase Stefan velocity [m/s]
        interface_diffusion_flux_liquid: liquid phase fuel interface diffusion flux [kg/m^2/s]
        interface_diffusion_flux_gas: gas phase fuel interface diffusion flux [kg/m^2/s]
        evaporation_rate: evaporation rate [kg/s]
        heat_flux_liquid: liquid phase heat transfer rate [W/m^2]
        heat_flux_gas: gas phase heat transfer rate [W/m^2]
        evaporation_heat_flux: evaporation heat transfer rate [W/m^2]
    """
    temperature: float = 0.0
    stefan_velocity_liquid: float = 0.0
    stefan_velocity_gas: float = 0.0
    old_stefan_velocity_liquid: float = 0.0
    old_stefan_velocity_gas: float = 0.0
    interface_diffusion_flux_liquid: np.ndarray = field(default_factory=lambda: np.zeros(40))
    interface_diffusion_flux_gas: np.ndarray = field(default_factory=lambda: np.zeros(40))
    evaporation_rate:  np.ndarray = field(default_factory=lambda: np.zeros(40))
    heat_flux_liquid: float = 0.0
    heat_flux_gas: float = 0.0
    evaporation_heat_flux: float = 0.0

class Surface:
    """surface management class
    
    used to manage the state parameters and boundary conditions of the droplet surface, including:
    1. gas phase surface state
    2. liquid phase surface state
    3. surface temperature
    4. surface heat flux
    5. surface mass flux
    """
    
    def __init__(self, grid, gas_surface, liquid_surface, standard_deviation, I_ini):
        """initialize the surface object
        
        Args:
            grid: grid object
            gas_surface: gas phase surface state
            liquid_surface: liquid phase surface state
            standard_deviation: standard deviation for surface (will be renamed to liquid_standard_deviation)
            I_ini: initial I value for surface
        """
        self.grid = grid
        self.gas_surface = gas_surface
        self.liquid_surface = liquid_surface
        self.temperature = None
        self.heat_flux = None
        self.mass_flux = None
        self.liquid_standard_deviation = standard_deviation
        self.alpha = None
        self.beta_l = None
        self.I_ini = I_ini
        self.liquid_mean = None
        # calculate split anchor: first value of liquid phase surface mole fraction
        self.split_anchor_y = self.liquid_surface.composition[0]
        self.split_anchor_x = None
        # parameters for gas phase continuous distribution
        self.c_k = None
        self.c_ek = None
        self.beta_g = None
        self.gas_coeff = None
        self.gas_mean = None
        self.gas_standard_deviation = None
        self.state = SurfaceState(temperature=self.liquid_surface.temperature)
        self._precompute_indices()
        self.initialize_gas_surface_state()

    def _precompute_indices(self):
        """
        precompute and cache the commonly used component index mapping.
        
        mainly includes:
            - gas phase fuel component index (gas_fuel_indices)
            - gas phase non-fuel component index (gas_non_fuel_indices)
            - liquid phase component index with corresponding gas phase component (liquid_fuel_indices)
            - liquid phase component index without corresponding gas phase component (liquid_non_fuel_indices)
            - liquid phase to gas phase component mapping (species_liquid2gas)
        """
        self.species_names_gas = self.gas_surface.species_names
        self.species_names_liquid = self.liquid_surface.species_names
        self.species_liquid2gas = init_species_mapping(self.species_names_gas)
        self.gas_fuel_indices = np.array([v for v in self.species_liquid2gas.values() if v is not None])
        self.gas_non_fuel_indices = np.array([i for i in range(len(self.species_names_gas)) 
                                        if i not in self.gas_fuel_indices])
        self.liquid_fuel_indices = np.array([i for i, v in self.species_liquid2gas.items() if v is not None])
        self.liquid_non_fuel_indices = np.array([i for i, v in self.species_liquid2gas.items() if v is None])

    def initialize_gas_surface_state(self):
        """
        initialize the gas phase surface state.
        set the gas phase surface state to be consistent with the liquid phase surface temperature, pressure and component balance.
        """
        self.gas_surface.TPX = (
            self.liquid_surface.temperature,
            self.liquid_surface.pressure,
            self.calculate_gas_surface_continuous_distribution()
        )

    def calculate_gas_surface_composition(self):
        """
        calculate the gas-liquid interface component balance, return the normalized gas phase component mole fraction.
        
        Returns:
            np.ndarray: gas phase surface component mole fraction (normalized)
        """
        gas_mole_fractions = np.array(self.gas_surface.X)
        fuel_partial_pressure = np.zeros_like(gas_mole_fractions)
        vapor_pressure = self.liquid_surface.vapor_pressure_ij()
        activity_coefficient = self.liquid_surface.activity_coefficient
        composition_liquid = self.liquid_surface.composition

        # only assign values to components with mapping
        fuel_partial_pressure[self.gas_fuel_indices] = (
            vapor_pressure[self.liquid_fuel_indices] *
            activity_coefficient[self.liquid_fuel_indices] *
            composition_liquid[self.liquid_fuel_indices]
        )
        gas_mole_fractions[self.gas_fuel_indices] = fuel_partial_pressure[self.gas_fuel_indices] / self.gas_surface.P

        # non-fuel component normalization
        fuel_mole_fraction_sum = np.sum(gas_mole_fractions[self.gas_fuel_indices])
        non_fuel_mole_fraction_total = 1.0 - fuel_mole_fraction_sum
        if non_fuel_mole_fraction_total > 0:
            current_non_fuel_total = np.sum(gas_mole_fractions[self.gas_non_fuel_indices])
            if current_non_fuel_total > 0:
                scale_factor = non_fuel_mole_fraction_total / current_non_fuel_total
                gas_mole_fractions[self.gas_non_fuel_indices] *= scale_factor
        # final normalization
        return gas_mole_fractions / np.sum(gas_mole_fractions)
    
    def calculate_gas_surface_continuous_distribution(self):
        """
        计算气相表面组分（使用连续分布方法）。
        
        该函数包括：
        1. calculate_equivalent_liquid_continuous_distribution: 计算等效液相连续分布
        2. calculate_equivalent_gas_continuous_distribution: 计算等效气相连续分布
        3. calculate_equivalent_liquid_and_standard_deviation_change: 计算气相离散分布
        
        返回:
            np.ndarray: 气相表面组分
        """
        # 计算等效液相连续分布
        self.calculate_equivalent_liquid_continuous_distribution()
        
        # 计算等效气相连续分布
        self.calculate_equivalent_gas_continuous_distribution()
        
        # 计算气相离散分布
        gas_mole_fractions = np.array(self.gas_surface.X)
        gas_mole_fractions[1],gas_mole_fractions[10] = self.calculate_discrete_gas_distribution()
        fuel_mole_fraction_sum = np.sum(gas_mole_fractions[self.gas_fuel_indices])
        non_fuel_mole_fraction_total = 1.0 - fuel_mole_fraction_sum
        if non_fuel_mole_fraction_total > 0:
            current_non_fuel_total = np.sum(gas_mole_fractions[self.gas_non_fuel_indices])
            if current_non_fuel_total > 0:
                scale_factor = non_fuel_mole_fraction_total / current_non_fuel_total
                gas_mole_fractions[self.gas_non_fuel_indices] *= scale_factor
        # final normalization
        return gas_mole_fractions / np.sum(gas_mole_fractions)
    
    def calculate_equivalent_liquid_continuous_distribution(self):
        """
        计算等效液相连续分布。
        
        该函数：
        1. 基于 composition * molecular_weights 计算平均分子量
        2. 计算 alpha = (liquid_mean - I_ini)^2 / liquid_standard_deviation^2
        3. 计算 beta_l = (liquid_mean - I_ini) / liquid_standard_deviation^2
        4. 使用三参数伽马分布计算 split_anchor_x
        
        返回:
            np.ndarray: 等效液相连续分布
        """
        # 计算平均分子量: composition * molecular_weights
        composition = self.liquid_surface.composition
        molecular_weights = liquid_para.MOLECULAR_WEIGHTS
        self.liquid_mean = np.sum(composition * molecular_weights)
        
        # 计算 alpha 和 beta_l
        # alpha = (Ī - x₀)² / σ²
        # beta_l = (Ī - x₀) / σ²
        # 其中 Ī 是 liquid_mean, x₀ 是 I_ini, σ² 是 liquid_standard_deviation²
        if self.liquid_standard_deviation is not None and self.liquid_standard_deviation != 0.0:
            if self.I_ini is not None:
                diff = self.liquid_mean - self.I_ini
                sigma_squared = self.liquid_standard_deviation ** 2
                self.alpha = (diff ** 2) / sigma_squared
                self.beta_l = diff / sigma_squared
                
                # 使用三参数伽马分布计算 split_anchor_x
                # 找到 split_anchor_x 使得 F(split_anchor_x) = split_anchor_y
                # 其中 F 是 Gamma(alpha, 1/beta_l, I_ini) 的累积分布函数
                # 使用 scipy.stats.gamma.ppf (分位数函数，累积分布函数的逆函数)
                if self.alpha > 0 and self.beta_l != 0:
                    # 伽马分布参数:
                    # a = alpha (形状参数)
                    # scale = 1/beta_l (尺度参数)
                    # loc = I_ini (位置参数，下界)
                    self.split_anchor_x = stats.gamma.ppf(
                        self.split_anchor_y,
                        a=self.alpha,
                        scale=1.0/self.beta_l,
                        loc=self.I_ini
                    )
                else:
                    self.split_anchor_x = self.I_ini
            else:
                # 如果 I_ini 为 None，将 alpha 和 beta_l 设为 0
                self.alpha = 0.0
                self.beta_l = 0.0
                self.split_anchor_x = None
        else:
            # 如果 liquid_standard_deviation 为 None 或 0，将 alpha 和 beta_l 设为 0
            self.alpha = 0.0
            self.beta_l = 0.0
            self.split_anchor_x = None
    
    def calculate_equivalent_gas_continuous_distribution(self):
        """
        计算等效气相连续分布。
        
        该函数：
        1. 从 pop_S01_VP.npy 文件加载参数
        2. 基于界面温度对参数进行插值
        3. 从插值结果中提取 c_k 和 c_ek
        4. 计算气相伽马分布的均值和标准差: 
            mean = alpha * scale + loc = alpha * (1/beta_g) + I_ini,
            sigma = scale * sqrt(shape) = (1/beta_g) * sqrt(alpha),
            其中位置参数 I_ini影响均值，不影响标准差
        返回:
            np.ndarray: 等效气相连续分布
        """
        # 获取界面温度
        interface_temperature = self.liquid_surface.temperature
        
        # 使用包资源从 npy 文件加载参数
        # 从 src.core 包访问文件
        resource_path = files('src.core') / 'pop_S01_VP.npy'
        popt = np.load(str(resource_path))
        
        # 创建 xp 数组: [275, 276, ..., 549]
        xp = np.arange(275) + 275
        
        # 初始化结果数组
        popt_VP = np.zeros((4, 2))
        
        # 对每个 i 和 j 进行插值
        for i in range(4):
            for j in range(2):
                popt_VP[i, j] = np.interp(interface_temperature, xp, popt[i, :, j])
        
        # 从第一行提取 c_k 和 c_ek 并保存为属性
        self.c_k, self.c_ek = popt_VP[0, :]
        
        # 确保 alpha 和 beta_l 已计算（从液相连续分布）
        # 这些应该在调用 calculate_equivalent_liquid_continuous_distribution() 时已经计算
        if self.alpha is None or self.beta_l is None:
            raise ValueError("alpha 和 beta_l 必须首先计算。请先调用 calculate_equivalent_liquid_continuous_distribution()。")
        
        # 为清晰起见，赋值变量
        beta_l = self.beta_l
        alpha_l = self.alpha
        x_o = self.I_ini
        
        # 检查收敛条件
        if beta_l <= self.c_ek:
            raise ValueError(f"beta_l ({beta_l}) 必须大于 c_ek ({self.c_ek})")
        
        # 计算 beta_g 和 gas_coeff
        # beta_g = beta_l - B(T)，其中 B(T) = c_ek
        self.beta_g = beta_l - self.c_ek
        
        # C = [A(T) * e^(B(T)*x_o) / p] * (beta_l / beta_g)^alpha
        # 其中 A(T) = c_k, B(T) = c_ek, p = pressure
        pressure = self.gas_surface.P  # 或 self.liquid_surface.pressure
        self.gas_coeff = (self.c_k * np.exp(self.c_ek * x_o) / pressure) * (beta_l / self.beta_g)**alpha_l
        # 计算气相伽马分布的均值
        # 对于 Gamma(alpha, scale=1/beta_g, loc=I_ini):
        # 均值 = alpha * scale + loc = alpha * (1/beta_g) + I_ini
        self.gas_mean = self.alpha * (1.0 / self.beta_g) + x_o
        # 计算气相伽马分布的标准差
        # 对于 Gamma(alpha, scale=1/beta_g, loc=I_ini):
        # 标准差 = scale * sqrt(shape) = (1/beta_g) * sqrt(alpha)
        self.gas_standard_deviation = (1.0 / self.beta_g) * np.sqrt(self.alpha)
    
    def calculate_discrete_gas_distribution(self):
        """
        计算离散气相分布。
        
        该函数使用分位数原理将连续气相分布反演为离散气相分布。
        
        步骤：
        1. 使用液相表面的离散摩尔分数计算分位数 P_i
        2. 使用液相伽马分布的逆CDF找到分子量域边界 I_i,cut
        3. 对连续气相分布进行积分，计算离散组分的摩尔分数
        
        返回:
            tuple: (x_g_0, x_g_9) 第0个和第9个组分的摩尔分数
        """
        # 确保必要的参数已计算
        if self.alpha is None or self.beta_l is None or self.beta_g is None or self.gas_coeff is None:
            raise ValueError("必须首先计算 alpha, beta_l, beta_g 和 gas_coeff")
        
        # 确保 split_anchor_x 已计算（在 calculate_equivalent_liquid_continuous_distribution 中计算）
        if self.split_anchor_x is None:
            raise ValueError("split_anchor_x 必须首先计算。请先调用 calculate_equivalent_liquid_continuous_distribution()。")
        
        # 使用已计算的分割点
        I_cut = self.split_anchor_x
        
        # 获取组分的分子量
        molecular_weights = liquid_para.MOLECULAR_WEIGHTS
        I_0 = molecular_weights[0]  # 第0个组分的分子量
        I_9 = molecular_weights[9]  # 第9个组分的分子量
        
        # 计算积分 ∫_{I_a}^{I_b} I f_{g,s}(I) dI
        # 对于第0个组分: [I_ini, I_cut]
        # 对于第9个组分: [I_cut, ∞]
        
        x_o = self.I_ini
        
        # 计算第0个组分的积分: ∫_{I_ini}^{I_cut} I f_{g,s}(I) dI
        y_a_0 = self.I_ini - x_o  # = 0
        y_b_0 = I_cut - x_o
        
        # 使用正则化不完全伽马函数计算积分
        # ∫_{I_a}^{I_b} I g(I) dI = x_o [P(α, β_g y_b) - P(α, β_g y_a)] + (α/β_g) [P(α+1, β_g y_b) - P(α+1, β_g y_a)]
        # 其中 P(a, x) = gammainc(a, x) 是正则化不完全伽马函数
        integral_0 = (
            x_o * (gammainc(self.alpha, self.beta_g * y_b_0) - gammainc(self.alpha, self.beta_g * y_a_0)) +
            (self.alpha / self.beta_g) * (gammainc(self.alpha + 1, self.beta_g * y_b_0) - gammainc(self.alpha + 1, self.beta_g * y_a_0))
        )
        
        # 计算第9个组分的积分: ∫_{I_cut}^{∞} I f_{g,s}(I) dI
        # 对于 [I_cut, ∞]，使用总积分减去 [I_ini, I_cut] 的积分
        # 总积分 = ∫_{I_ini}^{∞} I f_{g,s}(I) dI
        # 对于归一化的伽马分布 Gamma(alpha, scale=1/beta_g, loc=I_ini)
        # 根据公式: ∫_{I_ini}^{∞} I g(I) dI = x_o * [1 - P(α, 0)] + (α/β_g) * [1 - P(α+1, 0)]
        # 由于 P(α, 0) = 0，所以总积分 = x_o + (α/β_g) = gas_mean
        total_integral = x_o + (self.alpha / self.beta_g)
        integral_9 = total_integral - integral_0
        
        # 计算离散组分的摩尔分数
        # x_{g,i} = C * (∫ I f_{g,s}(I) dI) / I_i
        x_g_0 = self.gas_coeff * integral_0 / I_0
        x_g_9 = self.gas_coeff * integral_9 / I_9
        
        return x_g_0, x_g_9
