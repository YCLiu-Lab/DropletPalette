"""
liquid utils module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.

This module provides various utility functions for liquid mixtures (liquid, liquid_array) calculations, including:

1. liquid setting validation functions:
   - validate_composition: validate the validity of a single component
   - validate_pressure: validate the validity of pressure
   - validate_temperature: validate the validity of temperature

2. liquid_array setting validation functions:
   - validate_composition_array: validate the validity of composition array
   - validate_composition_array_with_message: validate the validity of composition array with detailed error message
   - validate_pressure_array: validate the validity of pressure array
   - validate_temperature_array: validate the validity of temperature array

3. supercritical species checking functions:
   - find_supercritical_species: check and report supercritical species

4. utility functions:
   - normalize: normalize the composition array
   - get_matrix_result: get the matrix result according to the index
"""

import numpy as np
import numba
from typing import Union, Optional
from .liquid_para import TC, COM_TOLERANCE, SPECIES_NAMES


# 常量定义
PRESSURE_MIN = 0.0
TEMPERATURE_MIN = 0.0
NUM_SPECIES = 40    # 组分数量
# 1. liquid设置验证函数：

@numba.njit(cache=True)
def _validate_composition_impl(composition: np.ndarray) -> None:
    """验证组分的实现函数"""
    if composition.size != NUM_SPECIES:
        raise ValueError("Invalid composition size")
    
    if np.any(composition < 0):
        for i in range(composition.size):
            if composition[i] < 0:
                raise ValueError("Negative composition value")
    

@numba.njit(cache=True)
def _validate_pressure_impl(pressure: float) -> None:
    """验证压强的实现函数"""
    if pressure <= PRESSURE_MIN:
        raise ValueError("Invalid pressure")

@numba.njit(cache=True)
def _validate_temperature_impl(temperature: float) -> None:
    """验证温度的实现函数"""
    if temperature <= TEMPERATURE_MIN:
        raise ValueError("Invalid temperature")

# 预热numba函数
def _warmup_numba():
    """预热numba函数以避免首次调用的编译开销"""
    try:
        test_comp = np.ones(NUM_SPECIES) / NUM_SPECIES
        _validate_composition_impl(test_comp)
        _validate_pressure_impl(101325.0)
        _validate_temperature_impl(300.0)
    except:
        pass

# 在模块导入时预热
_warmup_numba()

def validate_composition(composition: np.ndarray) -> None:
    """验证组分摩尔含量的有效性"""
    try:
        _validate_composition_impl(composition)
    except ValueError as e:
        if str(e) == "Invalid composition size":
            raise ValueError(f"组分数组维度错误: 期望{NUM_SPECIES}, 实际{composition.size}")
        elif str(e) == "Negative composition value":
            for i, x in enumerate(composition):
                if x < 0:
                    if x < -1e-3:
                        raise ValueError(f"存在负的摩尔分数: 索引{i}, 值{x}")
                    else:
                        x = 0


def validate_pressure(pressure: float) -> None:
    """验证压强的有效性"""
    try:
        _validate_pressure_impl(pressure)
    except ValueError:
        raise ValueError(f"压强必须大于{PRESSURE_MIN}Pa，当前值: {pressure}Pa")

def validate_temperature(temperature: float) -> None:
    """验证温度的有效性"""
    try:
        _validate_temperature_impl(temperature)
    except ValueError:
        raise ValueError(f"温度必须大于{TEMPERATURE_MIN}K，当前值: {temperature}K")



# 2. liquid_array验证函数：

@numba.njit(cache=True)
def validate_temperature_array(temperature: np.ndarray) -> None:
    """验证温度数组是否有效，如果无效则直接抛出异常
    
    参数:
        temperature (np.ndarray): 温度数组
        
    异常:
        ValueError: 当温度数组存在无效值时抛出
    """
    if np.any(temperature <= 0):
        raise ValueError("温度必须大于0，存在无效值")

@numba.njit(cache=True)
def validate_pressure_array(pressure: np.ndarray) -> None:
    """验证压强数组是否有效，如果无效则直接抛出异常
    
    参数:
        pressure (np.ndarray): 压强数组
        
    异常:
        ValueError: 当压强数组存在无效值时抛出
    """
    if np.any(pressure <= 0):
        raise ValueError("压强必须大于0，存在无效值")

@numba.njit(cache=True)
def validate_composition_array(composition: np.ndarray) -> None:
    """验证组分数组是否有效，如果无效则直接抛出异常
    
    参数:
        composition (np.ndarray): 组分数组，形状为(n, 40)
        
    异常:
        ValueError: 当组分数组存在无效值时抛出
    """
    # 验证数组维度
    if composition.ndim != 2:
        raise ValueError("Invalid array dimension")
    if composition.shape[1] != NUM_SPECIES:
        raise ValueError("Invalid array shape")
        
    # 验证非负性
    if np.any(composition < 0):
        raise ValueError("Negative composition values")
        
    # 验证每行的组分和接近1
    for i in range(composition.shape[0]):
        sum_comp = np.sum(composition[i])
        if abs(sum_comp - 1.0) > COM_TOLERANCE:
            raise ValueError("Sum of composition not 1")

def validate_composition_array_with_message(composition: np.ndarray) -> None:
    """带有详细错误消息的组分数组验证函数"""
    try:
        validate_composition_array(composition)
    except ValueError as e:
        if str(e) == "Invalid array dimension":
            raise ValueError(f"组分数组必须是二维数组，当前维度: {composition.ndim}")
        elif str(e) == "Invalid array shape":
            raise ValueError(f"组分数组第二维必须是{NUM_SPECIES}，当前形状: {composition.shape}")
        elif str(e) == "Negative composition values":
            raise ValueError("组分必须大于等于0，存在无效值")
        elif str(e) == "Sum of composition not 1":
            for i in range(composition.shape[0]):
                sum_comp = np.sum(composition[i])
                if abs(sum_comp - 1.0) > COM_TOLERANCE:
                    raise ValueError(f"第{i}行的组分和不为1: {sum_comp}, 误差{abs(sum_comp-1.0)} > {COM_TOLERANCE}")
        else:
            raise


# 3. 超临界组分检查函数：

@numba.njit(cache=True)
def _find_supercritical_species_impl(composition: np.ndarray, temperature: float, tc: np.ndarray) -> None:
    """超临界检查的实现函数"""
    # 使用向量化操作找出所有超临界组分
    mask = (composition > 0) & (temperature > tc)
    if np.any(mask):
        # 只在有超临界组分时遍历以获取具体信息
        for i in range(len(composition)):
            if mask[i]:
                print("Warning: Supercritical species detected")

def find_supercritical_species(composition: np.ndarray, temperature: float) -> None:
    """查找超临界组分，如果存在则抛出警告"""
    _find_supercritical_species_impl(composition, temperature, TC)
    # 如果有超临界组分，生成详细的警告信息
    mask = (composition > 0) & (temperature > TC)
    if np.any(mask):
        for i in range(len(composition)):
            if mask[i]:
                print(f"警告: 组分 {SPECIES_NAMES[i]} 处于超临界状态 (T = {temperature:.2f}K > Tc = {TC[i]:.2f}K)")

# 4. 工具函数：

@numba.njit(cache=True)
def normalize(composition: np.ndarray) -> np.ndarray:
    """归一化组分数组
    
    Args:
        composition: 组分数组，可以是摩尔分数或质量分数
        
    Returns:
        np.ndarray: 归一化后的组分数组

    """
    total = np.sum(composition)
    if total > 0:
        return composition / total
    else:
        return np.zeros_like(composition)

# 工具函数-根据索引返回矩阵的相应部分   
def get_matrix_result(matrix: np.ndarray, 
                     i: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    根据索引返回矩阵的相应部分。如果索引为None，则返回整个矩阵。
    
    参数:
        matrix (np.ndarray): 输入矩阵，形状为(40,)
        i (Optional[int]): 索引值，如果为None则返回整个矩阵
        
    返回:
        Union[float, np.ndarray]: 
            - 当i为None时，返回整个矩阵
            - 当i为有效索引时，返回对应位置的元素
            
    异常:
        IndexError: 当索引超出矩阵范围时
    """
    if i is None:
        return matrix
        
    if i < 0 or i >= len(matrix):
        raise IndexError("索引%d超出矩阵范围[0, %d]" % (i, len(matrix)-1))
        
    return matrix[i]