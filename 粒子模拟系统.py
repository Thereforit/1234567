"""
粒子模拟系统 - Particle Simulation System
完整版代码，包含所有功能模块
作者：AI Assistant
版本：2.0.0
"""

import pygame
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional, Any, Callable
import colorsys
import time
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from collections import defaultdict


# ==================== 枚举类型定义 ====================

class RenderMode(Enum):
    """渲染模式枚举"""
    PARTICLES = "particles"  # 仅粒子
    TRAILS = "trails"  # 轨迹
    DENSITY = "density"  # 密度场
    VELOCITY = "velocity"  # 速度场
    PRESSURE = "pressure"  # 压力场
    COMPOSITE = "composite"  # 复合模式
    HEATMAP = "heatmap"  # 热力图
    VECTOR = "vector"  # 矢量场


class ParticleType(Enum):
    """粒子类型枚举"""
    ELECTRON = "electron"  # 电子
    PROTON = "proton"  # 质子
    NEUTRON = "neutron"  # 中子
    PHOTON = "photon"  # 光子
    NEUTRAL = "neutral"  # 中性粒子
    CUSTOM = "custom"  # 自定义


class ForceFieldType(Enum):
    """力场类型枚举"""
    UNIFORM = "uniform"  # 均匀力场
    VORTEX = "vortex"  # 漩涡场
    RADIAL = "radial"  # 径向场
    NOISE = "noise"  # 噪声场
    GRAVITY_WELL = "gravity_well"  # 重力井


class ConstraintType(Enum):
    """约束类型枚举"""
    CIRCLE = "circle"  # 圆形约束
    RECTANGLE = "rectangle"  # 矩形约束
    LINE = "line"  # 线约束
    POLYGON = "polygon"  # 多边形约束
    DAMPING = "damping"  # 阻尼边界


class IntegratorType(Enum):
    """积分器类型枚举"""
    EULER = "euler"  # 欧拉法
    VERLET = "verlet"  # Verlet积分
    RK4 = "rk4"  # 四阶龙格库塔


# ==================== 数据结构定义 ====================

@dataclass
class Vector2:
    """二维向量类"""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x / scalar, self.y / scalar)

    def dot(self, other: 'Vector2') -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Vector2') -> float:
        return self.x * other.y - self.y * other.x

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> 'Vector2':
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector2(0, 0)

    def rotate(self, angle: float) -> 'Vector2':
        """绕原点旋转角度（弧度）"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def distance_to(self, other: 'Vector2') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @classmethod
    def from_angle(cls, angle: float, magnitude: float = 1.0) -> 'Vector2':
        """从角度创建向量"""
        return cls(math.cos(angle) * magnitude, math.sin(angle) * magnitude)

    @classmethod
    def random(cls, min_mag: float = 0.0, max_mag: float = 1.0) -> 'Vector2':
        """创建随机向量"""
        angle = random.uniform(0, 2 * math.pi)
        mag = random.uniform(min_mag, max_mag)
        return cls.from_angle(angle, mag)


@dataclass
class Color:
    """颜色类"""
    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

    def to_rgb(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_hsv(self) -> Tuple[float, float, float]:
        return colorsys.rgb_to_hsv(self.r / 255, self.g / 255, self.b / 255)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: int = 255) -> 'Color':
        """从HSV创建颜色"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return cls(int(r * 255), int(g * 255), int(b * 255), a)

    @classmethod
    def random(cls, min_brightness: float = 0.5) -> 'Color':
        """生成随机颜色"""
        h = random.random()
        s = random.uniform(0.7, 1.0)
        v = random.uniform(min_brightness, 1.0)
        return cls.from_hsv(h, s, v)

    @classmethod
    def gradient(cls, value: float, colormap: str = "rainbow") -> 'Color':
        """根据值生成渐变颜色"""
        value = max(0.0, min(1.0, value))

        if colormap == "rainbow":
            h = value * 0.8  # 0.8避免紫色循环
            return cls.from_hsv(h, 0.8, 1.0)
        elif colormap == "jet":
            # Jet颜色映射
            if value < 0.125:
                r = 0
                g = 0
                b = 128 + 127 * (value / 0.125)
            elif value < 0.375:
                r = 0
                g = 127 + 128 * ((value - 0.125) / 0.25)
                b = 255
            elif value < 0.625:
                r = 127 + 128 * ((value - 0.375) / 0.25)
                g = 255
                b = 255 - 128 * ((value - 0.375) / 0.25)
            elif value < 0.875:
                r = 255
                g = 255 - 128 * ((value - 0.625) / 0.25)
                b = 0
            else:
                r = 255 - 127 * ((value - 0.875) / 0.125)
                g = 0
                b = 0
            return cls(int(r), int(g), int(b))
        elif colormap == "heat":
            # 热力图颜色
            r = 255
            g = 255 * (1 - value)
            b = 255 * (1 - value)
            return cls(int(r), int(g), int(b))
        else:  # viridis近似
            r = int(68 + 187 * value)
            g = int(1 + 254 * value ** 0.8)
            b = int(84 + 171 * value ** 1.2)
            return cls(r, g, b)


# ==================== 粒子类 ====================

class Particle:
    """粒子类"""

    def __init__(
            self,
            position: Tuple[float, float],
            velocity: Tuple[float, float] = (0, 0),
            radius: float = 3.0,
            mass: float = 1.0,
            color: Optional[Color] = None,
            particle_type: ParticleType = ParticleType.NEUTRAL,
            charge: float = 0.0,
            temperature: float = 300.0,
            lifetime: float = -1.0,  # -1表示无限寿命
            fixed: bool = False
    ):
        self.position = Vector2(*position)
        self.velocity = Vector2(*velocity)
        self.acceleration = Vector2(0, 0)
        self.force = Vector2(0, 0)  # 累积力

        self.radius = radius
        self.mass = mass
        self.color = color or Color.random()
        self.particle_type = particle_type
        self.charge = charge
        self.temperature = temperature
        self.lifetime = lifetime
        self.age = 0.0
        self.fixed = fixed  # 是否固定位置

        # 物理属性
        self.restitution = 0.9  # 弹性系数
        self.friction = 0.1  # 摩擦系数
        self.damping = 0.99  # 速度阻尼

        # 状态追踪
        self.trail: List[Vector2] = []
        self.max_trail_length = 50
        self.is_colliding = False
        self.cluster_id = -1  # 所属团簇ID

        # 热力学属性
        self.kinetic_energy = 0.0
        self.update_kinetic_energy()

    def update_kinetic_energy(self):
        """更新动能"""
        v2 = self.velocity.x ** 2 + self.velocity.y ** 2
        self.kinetic_energy = 0.5 * self.mass * v2

    def apply_force(self, force: Vector2):
        """施加力"""
        if not self.fixed:
            self.force += force

    def clear_forces(self):
        """清空累积力"""
        self.force = Vector2(0, 0)

    def update(self, dt: float):
        """更新粒子状态"""
        if self.fixed:
            return

        # 更新寿命
        self.age += dt
        if self.lifetime > 0 and self.age > self.lifetime:
            return False  # 粒子死亡

        # 计算加速度
        if self.mass > 0:
            self.acceleration = self.force / self.mass

        # 更新速度和位置（欧拉积分）
        self.velocity += self.acceleration * dt
        self.velocity *= self.damping  # 应用阻尼
        self.position += self.velocity * dt

        # 更新动能
        self.update_kinetic_energy()

        # 更新轨迹
        self.trail.append(Vector2(self.position.x, self.position.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

        # 清空力
        self.clear_forces()

        return True  # 粒子存活

    def distance_to(self, other: 'Particle') -> float:
        """计算到另一个粒子的距离"""
        return self.position.distance_to(other.position)

    def is_colliding_with(self, other: 'Particle') -> bool:
        """检测与另一个粒子的碰撞"""
        distance = self.distance_to(other)
        return distance < (self.radius + other.radius)

    def get_speed(self) -> float:
        """获取速度大小"""
        return self.velocity.magnitude()

    def get_momentum(self) -> Vector2:
        """获取动量"""
        return self.velocity * self.mass

    def get_color_by_speed(self, max_speed: float = 10.0) -> Color:
        """根据速度获取颜色"""
        speed = min(self.get_speed() / max_speed, 1.0)
        return Color.gradient(speed, "heat")

    def get_color_by_energy(self, max_energy: float = 100.0) -> Color:
        """根据能量获取颜色"""
        energy = min(self.kinetic_energy / max_energy, 1.0)
        return Color.gradient(energy, "jet")

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'position': self.position.to_tuple(),
            'velocity': self.velocity.to_tuple(),
            'radius': self.radius,
            'mass': self.mass,
            'color': self.color.to_tuple(),
            'type': self.particle_type.value,
            'charge': self.charge,
            'temperature': self.temperature,
            'age': self.age,
            'kinetic_energy': self.kinetic_energy
        }


# ==================== 力场系统 ====================

class ForceField:
    """力场基类"""

    def __init__(self, field_type: ForceFieldType, strength: float = 1.0):
        self.field_type = field_type
        self.strength = strength
        self.enabled = True

    def get_force_at(self, position: Vector2, particle: Optional[Particle] = None) -> Vector2:
        """获取在指定位置的力"""
        raise NotImplementedError

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'type': self.field_type.value,
            'strength': self.strength,
            'enabled': self.enabled
        }


class UniformForceField(ForceField):
    """均匀力场"""

    def __init__(self, direction: Tuple[float, float], strength: float = 1.0):
        super().__init__(ForceFieldType.UNIFORM, strength)
        self.direction = Vector2(*direction).normalize()

    def get_force_at(self, position: Vector2, particle: Optional[Particle] = None) -> Vector2:
        if not self.enabled:
            return Vector2(0, 0)
        return self.direction * self.strength

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['direction'] = self.direction.to_tuple()
        return data


class VortexForceField(ForceField):
    """漩涡力场"""

    def __init__(self, center: Tuple[float, float], strength: float = 1.0, radius: float = 100.0):
        super().__init__(ForceFieldType.VORTEX, strength)
        self.center = Vector2(*center)
        self.radius = radius

    def get_force_at(self, position: Vector2, particle: Optional[Particle] = None) -> Vector2:
        if not self.enabled:
            return Vector2(0, 0)

        # 计算到中心的向量
        vec = position - self.center
        distance = vec.magnitude()

        if distance < 1e-6:
            return Vector2(0, 0)

        # 漩涡力垂直于径向向量
        force_magnitude = self.strength * (1.0 - distance / self.radius)
        if force_magnitude < 0:
            return Vector2(0, 0)

        # 垂直向量（旋转90度）
        tangent = Vector2(-vec.y, vec.x).normalize()
        return tangent * force_magnitude

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['center'] = self.center.to_tuple()
        data['radius'] = self.radius
        return data


class RadialForceField(ForceField):
    """径向力场（引力/斥力）"""

    def __init__(self, center: Tuple[float, float], strength: float = 1.0, max_radius: float = 300.0):
        super().__init__(ForceFieldType.RADIAL, strength)
        self.center = Vector2(*center)
        self.max_radius = max_radius

    def get_force_at(self, position: Vector2, particle: Optional[Particle] = None) -> Vector2:
        if not self.enabled:
            return Vector2(0, 0)

        vec = position - self.center
        distance = vec.magnitude()

        if distance < 1e-6 or distance > self.max_radius:
            return Vector2(0, 0)

        # 归一化方向
        direction = vec.normalize()

        # 力的大小随距离衰减
        force_magnitude = self.strength * (1.0 - distance / self.max_radius)

        return direction * force_magnitude

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['center'] = self.center.to_tuple()
        data['max_radius'] = self.max_radius
        return data


class GravityWell(ForceField):
    """重力井（牛顿引力）"""

    def __init__(self, center: Tuple[float, float], strength: float = 100.0, radius: float = 200.0):
        super().__init__(ForceFieldType.GRAVITY_WELL, strength)
        self.center = Vector2(*center)
        self.radius = radius

    def get_force_at(self, position: Vector2, particle: Optional[Particle] = None) -> Vector2:
        if not self.enabled:
            return Vector2(0, 0)

        vec = self.center - position
        distance = vec.magnitude()

        if distance < 1e-6:
            return Vector2(0, 0)

        # 牛顿引力公式：F = G * m1 * m2 / r^2
        # 这里简化处理，假设粒子质量为1
        force_magnitude = self.strength / (distance * distance + 1.0)

        # 限制最大距离
        if distance > self.radius:
            force_magnitude *= (self.radius / distance) ** 2

        direction = vec.normalize()
        return direction * force_magnitude

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['center'] = self.center.to_tuple()
        data['radius'] = self.radius
        return data


class NoiseForceField(ForceField):
    """噪声力场"""

    def __init__(self, strength: float = 0.1, scale: float = 0.1, seed: int = None):
        super().__init__(ForceFieldType.NOISE, strength)
        self.scale = scale
        self.seed = seed or random.randint(0, 10000)
        random.seed(self.seed)

    def get_force_at(self, position: Vector2, particle: Optional[Particle] = None) -> Vector2:
        if not self.enabled:
            return Vector2(0, 0)

        # 使用柏林噪声或简单随机噪声
        # 这里使用简单随机噪声
        angle = random.uniform(0, 2 * math.pi)
        magnitude = random.uniform(0, self.strength)

        return Vector2.from_angle(angle, magnitude)

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['scale'] = self.scale
        data['seed'] = self.seed
        return data


# ==================== 约束系统 ====================

class Constraint:
    """约束基类"""

    def __init__(self, constraint_type: ConstraintType, elasticity: float = 0.9):
        self.constraint_type = constraint_type
        self.elasticity = elasticity  # 弹性系数
        self.enabled = True

    def apply(self, particle: Particle) -> bool:
        """应用约束，返回是否发生碰撞"""
        raise NotImplementedError

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'type': self.constraint_type.value,
            'elasticity': self.elasticity,
            'enabled': self.enabled
        }


class CircleConstraint(Constraint):
    """圆形约束"""

    def __init__(self, center: Tuple[float, float], radius: float, elasticity: float = 0.9):
        super().__init__(ConstraintType.CIRCLE, elasticity)
        self.center = Vector2(*center)
        self.radius = radius

    def apply(self, particle: Particle) -> bool:
        if not self.enabled:
            return False

        vec = particle.position - self.center
        distance = vec.magnitude()

        if distance <= self.radius - particle.radius:
            return False

        # 发生碰撞
        overlap = distance - (self.radius - particle.radius)
        if overlap > 0:
            # 计算法线方向
            normal = vec.normalize()

            # 将粒子推回约束内
            particle.position -= normal * overlap

            # 计算反射速度
            velocity_normal = particle.velocity.dot(normal)
            if velocity_normal < 0:  # 只处理向内的速度分量
                particle.velocity -= normal * velocity_normal * (1 + self.elasticity)

            return True

        return False

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['center'] = self.center.to_tuple()
        data['radius'] = self.radius
        return data


class RectangleConstraint(Constraint):
    """矩形约束"""

    def __init__(self, bounds: Tuple[float, float, float, float], elasticity: float = 0.9):
        super().__init__(ConstraintType.RECTANGLE, elasticity)
        self.x1, self.y1, self.x2, self.y2 = bounds
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1

    def apply(self, particle: Particle) -> bool:
        if not self.enabled:
            return False

        collided = False
        pos = particle.position
        radius = particle.radius

        # 检查左边界
        if pos.x - radius < self.x1:
            particle.position.x = self.x1 + radius
            particle.velocity.x = -particle.velocity.x * self.elasticity
            collided = True

        # 检查右边界
        if pos.x + radius > self.x2:
            particle.position.x = self.x2 - radius
            particle.velocity.x = -particle.velocity.x * self.elasticity
            collided = True

        # 检查上边界
        if pos.y - radius < self.y1:
            particle.position.y = self.y1 + radius
            particle.velocity.y = -particle.velocity.y * self.elasticity
            collided = True

        # 检查下边界
        if pos.y + radius > self.y2:
            particle.position.y = self.y2 - radius
            particle.velocity.y = -particle.velocity.y * self.elasticity
            collided = True

        return collided

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['bounds'] = (self.x1, self.y1, self.x2, self.y2)
        return data


class LineConstraint(Constraint):
    """线约束"""

    def __init__(self, point1: Tuple[float, float], point2: Tuple[float, float], elasticity: float = 0.9):
        super().__init__(ConstraintType.LINE, elasticity)
        self.point1 = Vector2(*point1)
        self.point2 = Vector2(*point2)

        # 计算线的方向
        self.direction = (self.point2 - self.point1).normalize()
        self.normal = Vector2(-self.direction.y, self.direction.x)
        self.length = self.point1.distance_to(self.point2)

    def apply(self, particle: Particle) -> bool:
        if not self.enabled:
            return False

        # 计算点到线段的最短距离
        # 使用向量投影
        v1 = particle.position - self.point1
        v2 = self.point2 - self.point1

        dot = v1.dot(v2)
        if dot < 0:
            # 最近点是point1
            closest = self.point1
        elif dot > self.length * self.length:
            # 最近点是point2
            closest = self.point2
        else:
            # 在线段上投影
            t = dot / (self.length * self.length)
            closest = self.point1 + v2 * t

        # 计算距离
        distance = particle.position.distance_to(closest)

        if distance <= particle.radius:
            # 发生碰撞
            overlap = particle.radius - distance
            if overlap > 0:
                # 计算法线方向
                normal = (particle.position - closest).normalize()

                # 将粒子推离线
                particle.position += normal * overlap

                # 反射速度
                velocity_normal = particle.velocity.dot(normal)
                if velocity_normal < 0:
                    particle.velocity -= normal * velocity_normal * (1 + self.elasticity)

                return True

        return False

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data['point1'] = self.point1.to_tuple()
        data['point2'] = self.point2.to_tuple()
        return data


# ==================== 碰撞检测系统 ====================

class CollisionDetector:
    """碰撞检测器"""

    def __init__(self):
        self.spatial_grid = {}
        self.cell_size = 50
        self.collision_pairs = []

    def clear_grid(self):
        """清空空间网格"""
        self.spatial_grid.clear()

    def add_particle_to_grid(self, particle: Particle, particle_id: int):
        """将粒子添加到空间网格"""
        cell_x = int(particle.position.x / self.cell_size)
        cell_y = int(particle.position.y / self.cell_size)
        cell_key = (cell_x, cell_y)

        if cell_key not in self.spatial_grid:
            self.spatial_grid[cell_key] = []

        self.spatial_grid[cell_key].append(particle_id)

    def get_nearby_particles(self, particle: Particle, particle_id: int) -> List[int]:
        """获取附近的粒子ID"""
        cell_x = int(particle.position.x / self.cell_size)
        cell_y = int(particle.position.y / self.cell_size)

        nearby_particles = []

        # 检查3x3的网格区域
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.spatial_grid:
                    nearby_particles.extend(self.spatial_grid[cell_key])

        # 移除自身
        if particle_id in nearby_particles:
            nearby_particles.remove(particle_id)

        return nearby_particles

    def detect_collisions(self, particles: List[Particle]) -> List[Tuple[int, int]]:
        """检测所有碰撞"""
        self.clear_grid()

        # 构建空间网格
        for i, particle in enumerate(particles):
            self.add_particle_to_grid(particle, i)

        # 检测碰撞
        collisions = []
        checked_pairs = set()

        for i, particle in enumerate(particles):
            if particle.fixed:
                continue

            # 获取附近粒子
            nearby_ids = self.get_nearby_particles(particle, i)

            for j in nearby_ids:
                if j <= i:  # 避免重复检测
                    continue

                pair_key = (min(i, j), max(i, j))
                if pair_key in checked_pairs:
                    continue

                other = particles[j]
                if other.fixed:
                    continue

                if particle.is_colliding_with(other):
                    collisions.append((i, j))
                    checked_pairs.add(pair_key)

        self.collision_pairs = collisions
        return collisions

    def resolve_collision(self, p1: Particle, p2: Particle):
        """解析两个粒子之间的碰撞"""
        # 计算碰撞法线
        normal = (p1.position - p2.position).normalize()

        # 计算相对速度
        relative_velocity = p1.velocity - p2.velocity

        # 计算法向速度
        velocity_along_normal = relative_velocity.dot(normal)

        # 如果粒子正在分离，不处理
        if velocity_along_normal > 0:
            return

        # 计算反弹系数（使用两个粒子弹性系数的平均值）
        e = min(p1.restitution, p2.restitution)

        # 计算冲量标量
        j = -(1 + e) * velocity_along_normal
        j /= (1 / p1.mass + 1 / p2.mass)

        # 应用冲量
        impulse = normal * j
        p1.velocity += impulse / p1.mass
        p2.velocity -= impulse / p2.mass

        # 修正位置，避免重叠
        distance = p1.distance_to(p2)
        overlap = (p1.radius + p2.radius) - distance

        if overlap > 0:
            correction = normal * (overlap * 0.5)
            p1.position += correction * (p2.mass / (p1.mass + p2.mass))
            p2.position -= correction * (p1.mass / (p1.mass + p2.mass))

        # 标记碰撞状态
        p1.is_colliding = True
        p2.is_colliding = True


# ==================== 渲染系统 ====================

class Renderer:
    """渲染器"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.render_mode = RenderMode.COMPOSITE
        self.show_trails = True
        self.show_vectors = False
        self.show_grid = False
        self.show_stats = True
        self.background_color = (10, 10, 20)  # 深蓝色背景
        self.particle_colormap = "velocity"  # velocity, energy, temperature, rainbow
        self.trail_length = 20
        self.vector_scale = 10.0

        # 创建离屏表面用于轨迹效果
        self.trail_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.trail_surface.set_alpha(50)  # 半透明

        # 字体初始化
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)

    def clear(self, surface: pygame.Surface):
        """清空屏幕"""
        surface.fill(self.background_color)
        # 清空轨迹表面
        self.trail_surface.fill((0, 0, 0, 0))

        # 绘制网格
        if self.show_grid:
            self.draw_grid(surface)

    def draw_grid(self, surface: pygame.Surface):
        """绘制网格"""
        grid_size = 50
        grid_color = (30, 30, 40)

        # 垂直线
        for x in range(0, self.width, grid_size):
            pygame.draw.line(surface, grid_color, (x, 0), (x, self.height), 1)

        # 水平线
        for y in range(0, self.height, grid_size):
            pygame.draw.line(surface, grid_color, (0, y), (self.width, y), 1)

    def get_particle_color(self, particle: Particle, max_speed: float = 10.0) -> Color:
        """根据设置获取粒子颜色"""
        if self.particle_colormap == "velocity":
            speed = min(particle.get_speed() / max_speed, 1.0)
            return Color.gradient(speed, "heat")
        elif self.particle_colormap == "energy":
            energy = min(particle.kinetic_energy / 100.0, 1.0)
            return Color.gradient(energy, "jet")
        elif self.particle_colormap == "temperature":
            temp = min(particle.temperature / 1000.0, 1.0)
            return Color.gradient(temp, "rainbow")
        elif self.particle_colormap == "charge":
            if particle.charge > 0:
                return Color(255, 100, 100)  # 红色表示正电荷
            elif particle.charge < 0:
                return Color(100, 100, 255)  # 蓝色表示负电荷
            else:
                return Color(200, 200, 200)  # 灰色表示中性
        else:  # rainbow
            hue = (particle.age * 0.1) % 1.0
            return Color.from_hsv(hue, 0.8, 1.0)

    def draw_particles(self, surface: pygame.Surface, particles: List[Particle]):
        """绘制粒子"""
        max_speed = max((p.get_speed() for p in particles), default=10.0)

        for particle in particles:
            # 获取颜色
            color = self.get_particle_color(particle, max_speed)

            # 绘制粒子
            pos = (int(particle.position.x), int(particle.position.y))
            radius = int(particle.radius)

            # 根据碰撞状态调整颜色
            if particle.is_colliding:
                # 碰撞时高亮显示
                highlight_color = (min(color.r + 100, 255),
                                   min(color.g + 100, 255),
                                   min(color.b + 100, 255))
                pygame.draw.circle(surface, highlight_color, pos, radius + 2)
                particle.is_colliding = False  # 重置碰撞状态

            pygame.draw.circle(surface, color.to_rgb(), pos, radius)

            # 绘制粒子边框
            pygame.draw.circle(surface, (255, 255, 255), pos, radius, 1)

            # 绘制速度向量
            if self.show_vectors and particle.get_speed() > 0.1:
                end_pos = (int(pos[0] + particle.velocity.x * self.vector_scale),
                           int(pos[1] + particle.velocity.y * self.vector_scale))
                pygame.draw.line(surface, (255, 255, 0), pos, end_pos, 2)
                # 绘制箭头
                angle = math.atan2(particle.velocity.y, particle.velocity.x)
                arrow_length = 5
                arrow_angle = math.pi / 6

                arrow1 = (end_pos[0] - arrow_length * math.cos(angle - arrow_angle),
                          end_pos[1] - arrow_length * math.sin(angle - arrow_angle))
                arrow2 = (end_pos[0] - arrow_length * math.cos(angle + arrow_angle),
                          end_pos[1] - arrow_length * math.sin(angle + arrow_angle))

                pygame.draw.line(surface, (255, 255, 0), end_pos, arrow1, 2)
                pygame.draw.line(surface, (255, 255, 0), end_pos, arrow2, 2)

    def draw_trails(self, surface: pygame.Surface, particles: List[Particle]):
        """绘制粒子轨迹"""
        if not self.show_trails:
            return

        # 在轨迹表面上绘制
        for particle in particles:
            if len(particle.trail) < 2:
                continue

            # 根据粒子速度获取轨迹颜色
            speed = min(particle.get_speed() / 10.0, 1.0)
            trail_color = Color.gradient(speed, "rainbow")
            trail_color = (*trail_color.to_rgb(), 100)  # 添加透明度

            # 绘制轨迹线
            points = [(int(p.x), int(p.y)) for p in particle.trail]
            if len(points) >= 2:
                pygame.draw.lines(self.trail_surface, trail_color, False, points, 2)

        # 将轨迹表面合成到主表面
        surface.blit(self.trail_surface, (0, 0))

    def draw_force_fields(self, surface: pygame.Surface, force_fields: List[ForceField]):
        """绘制力场可视化"""
        for field in force_fields:
            if not field.enabled:
                continue

            if isinstance(field, VortexForceField):
                # 绘制漩涡场
                center = (int(field.center.x), int(field.center.y))
                radius = int(field.radius)
                pygame.draw.circle(surface, (100, 100, 255, 100), center, radius, 2)

                # 绘制旋转箭头
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    start_x = center[0] + math.cos(rad) * radius * 0.7
                    start_y = center[1] + math.sin(rad) * radius * 0.7
                    end_x = center[0] + math.cos(rad + math.pi / 4) * radius * 0.8
                    end_y = center[1] + math.sin(rad + math.pi / 4) * radius * 0.8

                    pygame.draw.line(surface, (100, 200, 255),
                                     (int(start_x), int(start_y)),
                                     (int(end_x), int(end_y)), 2)

            elif isinstance(field, GravityWell):
                # 绘制重力井
                center = (int(field.center.x), int(field.center.y))
                radius = int(field.radius)

                # 绘制同心圆
                for r in [radius * 0.3, radius * 0.6, radius * 0.9]:
                    pygame.draw.circle(surface, (255, 100, 100, 50), center, int(r), 1)

                # 绘制指向中心的箭头
                for angle in range(0, 360, 60):
                    rad = math.radians(angle)
                    start_x = center[0] + math.cos(rad) * radius * 0.8
                    start_y = center[1] + math.sin(rad) * radius * 0.8

                    pygame.draw.line(surface, (255, 150, 150),
                                     (int(start_x), int(start_y)),
                                     center, 2)

    def draw_constraints(self, surface: pygame.Surface, constraints: List[Constraint]):
        """绘制约束"""
        for constraint in constraints:
            if not constraint.enabled:
                continue

            if isinstance(constraint, CircleConstraint):
                center = (int(constraint.center.x), int(constraint.center.y))
                radius = int(constraint.radius)
                pygame.draw.circle(surface, (100, 255, 100, 100), center, radius, 2)

            elif isinstance(constraint, RectangleConstraint):
                rect = pygame.Rect(constraint.x1, constraint.y1,
                                   constraint.width, constraint.height)
                pygame.draw.rect(surface, (100, 255, 100, 100), rect, 2)

            elif isinstance(constraint, LineConstraint):
                start = (int(constraint.point1.x), int(constraint.point1.y))
                end = (int(constraint.point2.x), int(constraint.point2.y))
                pygame.draw.line(surface, (100, 255, 100, 100), start, end, 2)

    def draw_statistics(self, surface: pygame.Surface, stats: Dict[str, Any]):
        """绘制统计信息"""
        if not self.show_stats:
            return

        y_offset = 10
        x_offset = 10

        # 半透明背景
        stats_bg = pygame.Surface((250, 150), pygame.SRCALPHA)
        stats_bg.fill((0, 0, 0, 180))
        surface.blit(stats_bg, (x_offset, y_offset))

        # 绘制统计信息
        stat_lines = [
            f"粒子数量: {stats.get('particle_count', 0)}",
            f"帧率: {stats.get('fps', 0):.1f}",
            f"系统动能: {stats.get('kinetic_energy', 0):.2f}",
            f"系统势能: {stats.get('potential_energy', 0):.2f}",
            f"总能量: {stats.get('total_energy', 0):.2f}",
            f"平均速度: {stats.get('avg_speed', 0):.2f}",
            f"碰撞次数: {stats.get('collisions', 0)}",
            f"模拟时间: {stats.get('simulation_time', 0):.2f}s"
        ]

        for i, line in enumerate(stat_lines):
            text = self.small_font.render(line, True, (255, 255, 255))
            surface.blit(text, (x_offset + 10, y_offset + 10 + i * 20))

        # 绘制控制说明
        controls = [
            "控制说明:",
            "空格键: 暂停/继续",
            "R键: 重置系统",
            "C键: 清除所有粒子",
            "1-6键: 切换渲染模式",
            "T键: 显示/隐藏轨迹",
            "V键: 显示/隐藏速度向量",
            "G键: 显示/隐藏网格",
            "鼠标左键: 添加粒子",
            "鼠标右键: 添加重力井"
        ]

        controls_bg = pygame.Surface((250, 200), pygame.SRCALPHA)
        controls_bg.fill((0, 0, 0, 180))
        surface.blit(controls_bg, (self.width - 260, 10))

        for i, line in enumerate(controls):
            text = self.small_font.render(line, True, (200, 200, 255))
            surface.blit(text, (self.width - 250, 15 + i * 20))

    def render(self, surface: pygame.Surface, particles: List[Particle],
               force_fields: List[ForceField], constraints: List[Constraint],
               stats: Dict[str, Any]):
        """主渲染函数"""
        self.clear(surface)

        # 根据渲染模式绘制
        if self.render_mode in [RenderMode.COMPOSITE, RenderMode.TRAILS]:
            self.draw_trails(surface, particles)

        if self.render_mode in [RenderMode.COMPOSITE, RenderMode.PARTICLES,
                                RenderMode.VELOCITY, RenderMode.PRESSURE]:
            self.draw_particles(surface, particles)

        # 绘制力场和约束
        self.draw_force_fields(surface, force_fields)
        self.draw_constraints(surface, constraints)

        # 绘制统计信息
        self.draw_statistics(surface, stats)

    def set_render_mode(self, mode: RenderMode):
        """设置渲染模式"""
        self.render_mode = mode

        # 更新相关设置
        if mode == RenderMode.TRAILS:
            self.show_trails = True
            self.trail_surface.set_alpha(30)  # 更透明的轨迹
        elif mode == RenderMode.VELOCITY:
            self.show_vectors = True
            self.particle_colormap = "velocity"
        elif mode == RenderMode.PRESSURE:
            self.particle_colormap = "energy"
        elif mode == RenderMode.DENSITY:
            self.show_grid = True


# ==================== 粒子发射器 ====================

class ParticleEmitter:
    """粒子发射器"""

    def __init__(
            self,
            position: Tuple[float, float],
            emission_rate: float = 10.0,  # 每秒发射粒子数
            velocity_range: Tuple[float, float] = (-2, 2),
            angle_range: Tuple[float, float] = (0, 2 * math.pi),
            radius_range: Tuple[float, float] = (2, 5),
            mass_range: Tuple[float, float] = (0.5, 1.5),
            color: Optional[Color] = None,
            particle_type: ParticleType = ParticleType.NEUTRAL,
            lifetime: float = -1.0,
            burst_mode: bool = False,
            burst_count: int = 10
    ):
        self.position = Vector2(*position)
        self.emission_rate = emission_rate
        self.velocity_range = velocity_range
        self.angle_range = angle_range
        self.radius_range = radius_range
        self.mass_range = mass_range
        self.color = color
        self.particle_type = particle_type
        self.lifetime = lifetime
        self.burst_mode = burst_mode
        self.burst_count = burst_count

        self.time_since_last_emission = 0.0
        self.enabled = True
        self.total_emitted = 0

    def update(self, dt: float) -> List[Particle]:
        """更新发射器，返回新发射的粒子列表"""
        if not self.enabled:
            return []

        new_particles = []
        self.time_since_last_emission += dt

        if self.burst_mode:
            # 爆发模式：一次发射多个粒子
            if self.time_since_last_emission >= 1.0 / self.emission_rate:
                for _ in range(self.burst_count):
                    new_particles.append(self.create_particle())
                self.time_since_last_emission = 0.0
                self.total_emitted += self.burst_count
        else:
            # 连续模式
            expected_particles = self.emission_rate * self.time_since_last_emission
            num_to_emit = int(expected_particles)

            for _ in range(num_to_emit):
                new_particles.append(self.create_particle())
                self.total_emitted += 1

            self.time_since_last_emission -= num_to_emit / self.emission_rate

        return new_particles

    def create_particle(self) -> Particle:
        """创建一个新粒子"""
        # 随机角度和速度
        angle = random.uniform(*self.angle_range)
        speed = random.uniform(*self.velocity_range)
        velocity = Vector2.from_angle(angle, speed)

        # 随机半径和质量
        radius = random.uniform(*self.radius_range)
        mass = random.uniform(*self.mass_range)

        # 添加随机位置偏移
        pos_offset = Vector2.random(0, radius)
        position = self.position + pos_offset

        return Particle(
            position=position.to_tuple(),
            velocity=velocity.to_tuple(),
            radius=radius,
            mass=mass,
            color=self.color,
            particle_type=self.particle_type,
            lifetime=self.lifetime
        )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'position': self.position.to_tuple(),
            'emission_rate': self.emission_rate,
            'velocity_range': self.velocity_range,
            'angle_range': self.angle_range,
            'radius_range': self.radius_range,
            'mass_range': self.mass_range,
            'color': self.color.to_tuple() if self.color else None,
            'particle_type': self.particle_type.value,
            'lifetime': self.lifetime,
            'burst_mode': self.burst_mode,
            'burst_count': self.burst_count,
            'enabled': self.enabled,
            'total_emitted': self.total_emitted
        }


# ==================== 统计系统 ====================

class Statistics:
    """统计系统"""

    def __init__(self):
        self.data = {
            'particle_count': 0,
            'kinetic_energy': 0.0,
            'potential_energy': 0.0,
            'total_energy': 0.0,
            'avg_speed': 0.0,
            'max_speed': 0.0,
            'temperature': 0.0,
            'pressure': 0.0,
            'collisions': 0,
            'simulation_time': 0.0,
            'fps': 0.0
        }

        self.history = defaultdict(list)
        self.max_history_length = 1000
        self.frame_count = 0
        self.last_update_time = time.time()
        self.total_collisions = 0

    def update(self, particles: List[Particle], collision_count: int, dt: float):
        """更新统计信息"""
        self.frame_count += 1

        # 计算当前时间
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:  # 每秒更新一次FPS
            self.data['fps'] = self.frame_count / (current_time - self.last_update_time)
            self.frame_count = 0
            self.last_update_time = current_time

        # 基本统计
        self.data['particle_count'] = len(particles)
        self.data['simulation_time'] += dt
        self.total_collisions += collision_count
        self.data['collisions'] = self.total_collisions

        if not particles:
            return

        # 计算速度和能量
        total_kinetic = 0.0
        total_speed = 0.0
        max_speed = 0.0

        for particle in particles:
            total_kinetic += particle.kinetic_energy
            speed = particle.get_speed()
            total_speed += speed
            max_speed = max(max_speed, speed)

        self.data['kinetic_energy'] = total_kinetic
        self.data['avg_speed'] = total_speed / len(particles)
        self.data['max_speed'] = max_speed

        # 计算温度（假设每个粒子有3个自由度）
        if len(particles) > 0:
            avg_kinetic = total_kinetic / len(particles)
            # T = (2/3) * (平均动能 / 玻尔兹曼常数)
            # 这里使用简化公式
            self.data['temperature'] = avg_kinetic * 100

        # 总能量（动能 + 势能）
        self.data['total_energy'] = self.data['kinetic_energy'] + self.data['potential_energy']

        # 保存历史数据
        for key in self.data:
            self.history[key].append(self.data[key])
            if len(self.history[key]) > self.max_history_length:
                self.history[key].pop(0)

    def get_data(self) -> Dict[str, Any]:
        """获取当前统计数据"""
        return self.data.copy()

    def get_history(self, key: str) -> List[float]:
        """获取历史数据"""
        return self.history.get(key, [])

    def reset(self):
        """重置统计数据"""
        self.data = {k: 0.0 for k in self.data}
        self.history.clear()
        self.total_collisions = 0
        self.frame_count = 0
        self.last_update_time = time.time()


# ==================== 主粒子系统 ====================

class ParticleSystem:
    """粒子系统主类"""

    def __init__(
            self,
            width: int = 800,
            height: int = 600,
            num_particles: int = 100,
            gravity: Tuple[float, float] = (0, 0.5),
            damping: float = 0.98,
            particle_radius: float = 3.0,
            particle_mass: float = 1.0,
            enable_boundary: bool = True,
            boundary_elasticity: float = 0.9,
            enable_particles_collision: bool = True,
            enable_attraction: bool = False,
            attraction_strength: float = 0.01
    ):
        # 系统参数
        self.width = width
        self.height = height
        self.gravity = Vector2(*gravity)
        self.damping = damping
        self.dt = 1.0 / 60.0  # 时间步长

        # 组件初始化
        self.particles: List[Particle] = []
        self.force_fields: List[ForceField] = []
        self.constraints: List[Constraint] = []
        self.emitters: List[ParticleEmitter] = []

        # 系统组件
        self.collision_detector = CollisionDetector()
        self.renderer = Renderer(width, height)
        self.statistics = Statistics()

        # 物理参数
        self.enable_particles_collision = enable_particles_collision
        self.enable_attraction = enable_attraction
        self.attraction_strength = attraction_strength

        # 系统状态
        self.running = False
        self.paused = False
        self.time_scale = 1.0
        self.max_particles = 5000

        # 初始化边界约束
        if enable_boundary:
            boundary = RectangleConstraint(
                bounds=(0, 0, width, height),
                elasticity=boundary_elasticity
            )
            self.constraints.append(boundary)

        # 初始化粒子
        self.initialize_particles(num_particles, particle_radius, particle_mass)

        # 初始化力场
        self.initialize_force_fields()

        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("粒子模拟系统")
        self.clock = pygame.time.Clock()

    def initialize_particles(self, num_particles: int, radius: float, mass: float):
        """初始化粒子"""
        self.particles.clear()

        for i in range(num_particles):
            # 随机位置
            x = random.uniform(radius, self.width - radius)
            y = random.uniform(radius, self.height - radius)

            # 随机速度
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)

            # 随机颜色
            color = Color.random()

            # 创建粒子
            particle = Particle(
                position=(x, y),
                velocity=(vx, vy),
                radius=radius,
                mass=mass,
                color=color
            )

            self.particles.append(particle)

    def initialize_force_fields(self):
        """初始化力场"""
        # 添加重力场
        gravity_field = UniformForceField(
            direction=self.gravity.to_tuple(),
            strength=self.gravity.magnitude()
        )
        self.force_fields.append(gravity_field)

        # 添加随机噪声场
        noise_field = NoiseForceField(strength=0.05)
        self.force_fields.append(noise_field)

    def add_particle(self, position: Tuple[float, float], **kwargs):
        """添加单个粒子"""
        if len(self.particles) >= self.max_particles:
            return None

        # 默认参数
        params = {
            'position': position,
            'velocity': (random.uniform(-2, 2), random.uniform(-2, 2)),
            'radius': random.uniform(2, 5),
            'mass': random.uniform(0.5, 1.5),
            'color': Color.random()
        }
        params.update(kwargs)

        particle = Particle(**params)
        self.particles.append(particle)
        return particle

    def add_force_field(self, force_field: ForceField):
        """添加力场"""
        self.force_fields.append(force_field)

    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.append(constraint)

    def add_emitter(self, emitter: ParticleEmitter):
        """添加发射器"""
        self.emitters.append(emitter)

    def add_gravity_well(self, position: Tuple[float, float], strength: float = 100.0, radius: float = 200.0):
        """添加重力井"""
        gravity_well = GravityWell(position, strength, radius)
        self.force_fields.append(gravity_well)
        return gravity_well

    def update(self):
        """更新粒子系统"""
        if self.paused:
            return

        # 更新发射器
        for emitter in self.emitters:
            new_particles = emitter.update(self.dt * self.time_scale)
            self.particles.extend(new_particles)

            # 限制最大粒子数
            if len(self.particles) > self.max_particles:
                self.particles = self.particles[-self.max_particles:]

        # 应用力场
        for particle in self.particles:
            # 重置力
            particle.clear_forces()

            # 应用重力
            particle.apply_force(self.gravity * particle.mass)

            # 应用其他力场
            for force_field in self.force_fields:
                force = force_field.get_force_at(particle.position, particle)
                particle.apply_force(force * particle.mass)

            # 应用粒子间作用力（如果启用）
            if self.enable_attraction:
                self.apply_particle_interactions(particle)

        # 更新粒子
        dead_particles = []
        for i, particle in enumerate(self.particles):
            if not particle.update(self.dt * self.time_scale):
                dead_particles.append(i)

        # 移除死亡的粒子
        for i in reversed(dead_particles):
            self.particles.pop(i)

        # 应用约束
        for constraint in self.constraints:
            for particle in self.particles:
                constraint.apply(particle)

        # 碰撞检测和处理
        collision_count = 0
        if self.enable_particles_collision:
            collisions = self.collision_detector.detect_collisions(self.particles)
            collision_count = len(collisions)

            for i, j in collisions:
                self.collision_detector.resolve_collision(self.particles[i], self.particles[j])

        # 更新统计信息
        self.statistics.update(self.particles, collision_count, self.dt * self.time_scale)

    def apply_particle_interactions(self, particle: Particle):
        """应用粒子间相互作用力"""
        # 简化的引力/斥力模型
        for other in self.particles:
            if other is particle:
                continue

            # 计算距离
            vec = other.position - particle.position
            distance = vec.magnitude()

            if distance < 1e-6:
                continue

            # 力的大小（与距离平方成反比）
            force_magnitude = self.attraction_strength / (distance * distance + 1.0)

            # 限制最大距离
            if distance > 100:
                force_magnitude *= (100 / distance) ** 2

            # 应用力
            direction = vec.normalize()
            force = direction * force_magnitude

            particle.apply_force(force)

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouseclick(event)

    def handle_keydown(self, event):
        """处理键盘事件"""
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
            print("模拟已" + ("暂停" if self.paused else "继续"))

        elif event.key == pygame.K_r:
            self.reset_system()
            print("系统已重置")

        elif event.key == pygame.K_c:
            self.particles.clear()
            print("所有粒子已清除")

        elif event.key == pygame.K_t:
            self.renderer.show_trails = not self.renderer.show_trails
            print("轨迹显示已" + ("开启" if self.renderer.show_trails else "关闭"))

        elif event.key == pygame.K_v:
            self.renderer.show_vectors = not self.renderer.show_vectors
            print("速度向量显示已" + ("开启" if self.renderer.show_vectors else "关闭"))

        elif event.key == pygame.K_g:
            self.renderer.show_grid = not self.renderer.show_grid
            print("网格显示已" + ("开启" if self.renderer.show_grid else "关闭"))

        elif event.key == pygame.K_s:
            self.save_screenshot()

        elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6]:
            mode_index = event.key - pygame.K_1
            modes = list(RenderMode)
            if mode_index < len(modes):
                self.renderer.set_render_mode(modes[mode_index])
                print(f"渲染模式切换为: {modes[mode_index].value}")

        elif event.key == pygame.K_UP:
            self.time_scale *= 1.2
            print(f"时间倍率: {self.time_scale:.2f}")

        elif event.key == pygame.K_DOWN:
            self.time_scale /= 1.2
            print(f"时间倍率: {self.time_scale:.2f}")

    def handle_mouseclick(self, event):
        """处理鼠标点击事件"""
        mouse_pos = pygame.mouse.get_pos()

        if event.button == 1:  # 左键：添加粒子
            self.add_particle(mouse_pos)
            print(f"在 {mouse_pos} 添加粒子")

        elif event.button == 3:  # 右键：添加重力井
            self.add_gravity_well(mouse_pos)
            print(f"在 {mouse_pos} 添加重力井")

        elif event.button == 2:  # 中键：添加漩涡场
            vortex = VortexForceField(mouse_pos, strength=0.5, radius=150)
            self.add_force_field(vortex)
            print(f"在 {mouse_pos} 添加漩涡场")

    def reset_system(self):
        """重置系统"""
        # 保留部分设置，重置粒子
        particle_count = len(self.particles)
        avg_radius = sum(p.radius for p in self.particles) / max(particle_count, 1)
        avg_mass = sum(p.mass for p in self.particles) / max(particle_count, 1)

        self.particles.clear()
        self.initialize_particles(particle_count, avg_radius, avg_mass)
        self.statistics.reset()

    def save_screenshot(self):
        """保存截图"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pygame.image.save(self.screen, filename)
        print(f"截图已保存: {filename}")

    def save_state(self, filename: str = "particle_system_state.json"):
        """保存系统状态"""
        state = {
            'system': {
                'width': self.width,
                'height': self.height,
                'gravity': self.gravity.to_tuple(),
                'damping': self.damping,
                'time_scale': self.time_scale
            },
            'particles': [p.to_dict() for p in self.particles],
            'force_fields': [f.to_dict() for f in self.force_fields],
            'constraints': [c.to_dict() for c in self.constraints],
            'statistics': self.statistics.get_data()
        }

        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"系统状态已保存: {filename}")

    def load_state(self, filename: str = "particle_system_state.json"):
        """加载系统状态"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)

            # 加载系统参数
            sys_info = state['system']
            self.width = sys_info['width']
            self.height = sys_info['height']
            self.gravity = Vector2(*sys_info['gravity'])
            self.damping = sys_info['damping']
            self.time_scale = sys_info['time_scale']

            # 清空现有粒子
            self.particles.clear()

            # 加载粒子
            for p_data in state['particles']:
                particle = Particle(
                    position=p_data['position'],
                    velocity=p_data['velocity'],
                    radius=p_data['radius'],
                    mass=p_data['mass'],
                    color=Color(*p_data['color']),
                    particle_type=ParticleType(p_data['type']),
                    charge=p_data['charge'],
                    temperature=p_data['temperature']
                )
                particle.age = p_data['age']
                particle.kinetic_energy = p_data['kinetic_energy']
                self.particles.append(particle)

            print(f"系统状态已加载: {filename}")
            print(f"加载了 {len(self.particles)} 个粒子")

        except Exception as e:
            print(f"加载状态失败: {e}")

    def run(self):
        """运行主循环"""
        self.running = True
        print("粒子模拟系统启动")
        print("控制说明:")
        print("  空格键: 暂停/继续")
        print("  R键: 重置系统")
        print("  C键: 清除所有粒子")
        print("  T键: 显示/隐藏轨迹")
        print("  V键: 显示/隐藏速度向量")
        print("  G键: 显示/隐藏网格")
        print("  S键: 保存截图")
        print("  1-6键: 切换渲染模式")
        print("  上下方向键: 调整时间倍率")
        print("  鼠标左键: 添加粒子")
        print("  鼠标右键: 添加重力井")
        print("  鼠标中键: 添加漩涡场")

        while self.running:
            # 处理事件
            self.handle_events()

            # 更新系统
            self.update()

            # 渲染
            stats = self.statistics.get_data()
            self.renderer.render(self.screen, self.particles,
                                 self.force_fields, self.constraints, stats)

            # 更新显示
            pygame.display.flip()

            # 控制帧率
            self.clock.tick(60)

        pygame.quit()
        print("模拟结束")

    def run_for_frames(self, num_frames: int):
        """运行指定帧数"""
        for _ in range(num_frames):
            if not self.running:
                break

            self.handle_events()
            self.update()

            # 渲染
            stats = self.statistics.get_data()
            self.renderer.render(self.screen, self.particles,
                                 self.force_fields, self.constraints, stats)

            pygame.display.flip()
            self.clock.tick(60)

    def get_statistics(self) -> Dict[str, List[float]]:
        """获取历史统计数据"""
        return dict(self.statistics.history)

    def calculate_radial_distribution(self, bin_count: int = 50, max_distance: float = 200.0) -> Dict[str, List[float]]:
        """计算径向分布函数"""
        bins = [0] * bin_count
        pair_count = 0

        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles[i + 1:], i + 1):
                distance = p1.distance_to(p2)
                if distance < max_distance:
                    bin_index = int(distance / max_distance * bin_count)
                    if bin_index < bin_count:
                        bins[bin_index] += 1
                        pair_count += 1

        # 归一化
        if pair_count > 0:
            for i in range(bin_count):
                r_inner = i * max_distance / bin_count
                r_outer = (i + 1) * max_distance / bin_count
                shell_volume = math.pi * (r_outer ** 2 - r_inner ** 2)
                density = pair_count / (self.width * self.height)
                bins[i] = bins[i] / (shell_volume * density * pair_count)

        r_values = [(i + 0.5) * max_distance / bin_count for i in range(bin_count)]

        return {
            'r': r_values,
            'g': bins
        }


# ==================== 高级模拟类 ====================

class FluidSimulation(ParticleSystem):
    """流体模拟（基于SPH简化版）"""

    def __init__(self, width=800, height=600, num_particles=500,
                 density=1.0, viscosity=0.01, surface_tension=0.1):
        super().__init__(width, height, num_particles)

        # 流体参数
        self.density = density
        self.viscosity = viscosity
        self.surface_tension = surface_tension
        self.rest_density = 1000.0
        self.gas_constant = 2000.0

        # SPH参数
        self.kernel_radius = 30.0
        self.pressure_stiffness = 500.0

        # 调整粒子参数
        for particle in self.particles:
            particle.radius = 4.0
            particle.mass = 0.1
            particle.damping = 0.99
            particle.color = Color(100, 150, 255)  # 流体蓝色

        # 添加边界
        self.constraints.clear()
        boundary = RectangleConstraint(
            bounds=(50, 50, width - 50, height - 50),
            elasticity=0.9
        )
        self.constraints.append(boundary)

        # 禁用吸引力
        self.enable_attraction = False

        # 设置渲染
        self.renderer.particle_colormap = "pressure"

    def update(self):
        """流体模拟更新"""
        if self.paused:
            return

        # 计算密度和压力
        densities = self.calculate_densities()
        pressures = self.calculate_pressures(densities)

        # 应用流体力学力
        for i, particle in enumerate(self.particles):
            # 重置力
            particle.clear_forces()

            # 应用重力
            particle.apply_force(self.gravity * particle.mass)

            # 应用压力梯度力
            pressure_force = self.calculate_pressure_force(i, densities, pressures)
            particle.apply_force(pressure_force)

            # 应用粘性力
            viscosity_force = self.calculate_viscosity_force(i)
            particle.apply_force(viscosity_force)

            # 应用表面张力
            tension_force = self.calculate_surface_tension_force(i)
            particle.apply_force(tension_force)

        # 更新粒子
        super().update()

    def calculate_densities(self) -> List[float]:
        """计算每个粒子的密度"""
        densities = [0.0] * len(self.particles)

        for i, p1 in enumerate(self.particles):
            density_sum = 0.0

            for j, p2 in enumerate(self.particles):
                if i == j:
                    continue

                r = p1.distance_to(p2)
                if r < self.kernel_radius:
                    # 使用平滑核函数
                    q = r / self.kernel_radius
                    if q <= 1.0:
                        w = (1 - q) ** 2
                        density_sum += w * p2.mass

            densities[i] = density_sum

        return densities

    def calculate_pressures(self, densities: List[float]) -> List[float]:
        """计算压力"""
        pressures = []
        for density in densities:
            # 理想气体状态方程简化版
            pressure = max(0.0, self.gas_constant * (density - self.rest_density))
            pressures.append(pressure)
        return pressures

    def calculate_pressure_force(self, index: int, densities: List[float], pressures: List[float]) -> Vector2:
        """计算压力梯度力"""
        p1 = self.particles[index]
        pressure_force = Vector2(0, 0)

        for j, p2 in enumerate(self.particles):
            if index == j:
                continue

            r_vec = p2.position - p1.position
            r = r_vec.magnitude()

            if 0 < r < self.kernel_radius:
                # 压力梯度
                q = r / self.kernel_radius
                if q <= 1.0:
                    # 核函数梯度
                    dw_dr = -2 * (1 - q) / self.kernel_radius

                    # 压力项
                    pressure_term = (pressures[index] + pressures[j]) / (2 * densities[index] * densities[j])
                    force_mag = -p2.mass * pressure_term * dw_dr

                    if r > 0:
                        direction = r_vec.normalize()
                        pressure_force += direction * force_mag

        return pressure_force

    def calculate_viscosity_force(self, index: int) -> Vector2:
        """计算粘性力"""
        p1 = self.particles[index]
        viscosity_force = Vector2(0, 0)

        for j, p2 in enumerate(self.particles):
            if index == j:
                continue

            r_vec = p2.position - p1.position
            r = r_vec.magnitude()

            if 0 < r < self.kernel_radius:
                q = r / self.kernel_radius
                if q <= 1.0:
                    # 速度差
                    vel_diff = p2.velocity - p1.velocity

                    # 粘性项
                    w = (1 - q) ** 2
                    viscosity_term = self.viscosity * p2.mass * w / self.density

                    viscosity_force += vel_diff * viscosity_term

        return viscosity_force

    def calculate_surface_tension_force(self, index: int) -> Vector2:
        """计算表面张力"""
        p1 = self.particles[index]
        color_field = Vector2(0, 0)
        color_laplacian = 0.0

        for j, p2 in enumerate(self.particles):
            if index == j:
                continue

            r_vec = p2.position - p1.position
            r = r_vec.magnitude()

            if 0 < r < self.kernel_radius:
                q = r / self.kernel_radius
                if q <= 1.0:
                    # 颜色场梯度
                    dw_dr = -2 * (1 - q) / self.kernel_radius
                    if r > 0:
                        direction = r_vec.normalize()
                        color_field += direction * dw_dr * p2.mass / self.density

                    # 颜色场拉普拉斯
                    color_laplacian += (1 - q) * p2.mass / self.density

        # 表面张力力
        if color_field.magnitude() > 0.001:
            tension_force = -self.surface_tension * color_field * color_laplacian
            return tension_force

        return Vector2(0, 0)


class PlasmaSimulation(ParticleSystem):
    """等离子体模拟"""

    def __init__(self, width=800, height=600, electrons=200, ions=100,
                 magnetic_field=(0, 0, 0.1), electric_field=(0.1, 0, 0)):
        super().__init__(width, height, electrons + ions)

        # 电磁场参数
        self.magnetic_field = Vector2(*magnetic_field[:2])  # 只取xy分量
        self.electric_field = Vector2(*electric_field[:2])

        # 清除现有粒子
        self.particles.clear()

        # 创建电子
        for _ in range(electrons):
            x = random.uniform(100, width - 100)
            y = random.uniform(100, height - 100)

            electron = Particle(
                position=(x, y),
                velocity=(random.uniform(-3, 3), random.uniform(-3, 3)),
                radius=2,
                mass=0.1,
                color=Color(100, 100, 255),  # 蓝色
                particle_type=ParticleType.ELECTRON,
                charge=-1.0,
                temperature=10000.0
            )
            self.particles.append(electron)

        # 创建离子
        for _ in range(ions):
            x = random.uniform(100, width - 100)
            y = random.uniform(100, height - 100)

            ion = Particle(
                position=(x, y),
                velocity=(random.uniform(-1, 1), random.uniform(-1, 1)),
                radius=4,
                mass=1.0,
                color=Color(255, 100, 100),  # 红色
                particle_type=ParticleType.PROTON,
                charge=1.0,
                temperature=1000.0
            )
            self.particles.append(ion)

        # 设置渲染
        self.renderer.particle_colormap = "charge"
        self.enable_particles_collision = False
        self.enable_attraction = True
        self.attraction_strength = 0.1

        # 添加电极
        self.electrodes = []

    def add_electrode(self, position: Tuple[float, float], voltage: float, radius: float = 20.0):
        """添加电极"""
        self.electrodes.append({
            'position': Vector2(*position),
            'voltage': voltage,
            'radius': radius
        })

        # 添加可视化约束
        electrode_constraint = CircleConstraint(position, radius, elasticity=0.0)
        self.constraints.append(electrode_constraint)

    def update(self):
        """等离子体模拟更新"""
        if self.paused:
            return

        # 应用电磁场力
        for particle in self.particles:
            # 重置力
            particle.clear_forces()

            # 电场力：F = q * E
            electric_force = self.electric_field * particle.charge
            particle.apply_force(electric_force)

            # 磁场力：F = q * (v × B)（洛伦兹力）
            if self.magnetic_field.magnitude() > 0:
                # 计算叉积 v × B（这里B只有z分量）
                v_cross_b = Vector2(
                    -particle.velocity.y * self.magnetic_field.magnitude(),
                    particle.velocity.x * self.magnetic_field.magnitude()
                )
                magnetic_force = v_cross_b * particle.charge
                particle.apply_force(magnetic_force)

            # 电极力
            for electrode in self.electrodes:
                r_vec = electrode['position'] - particle.position
                distance = r_vec.magnitude()

                if distance < electrode['radius']:
                    # 在电极内部，施加斥力
                    if distance > 0:
                        force_dir = r_vec.normalize()
                        force_mag = electrode['voltage'] * particle.charge / (distance * distance + 1.0)
                        particle.apply_force(force_dir * force_mag)

        # 更新粒子
        super().update()


# ==================== 实验类 ====================

class Experiment:
    """实验环境"""

    def __init__(self, title: str = "物理实验", description: str = "", width: int = 1000, height: int = 800):
        self.title = title
        self.description = description
        self.width = width
        self.height = height

        self.parameters = {}
        self.systems = []
        self.current_system = 0

        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 28)

        print(f"实验 '{title}' 已创建")
        print(f"描述: {description}")

    def add_parameter(self, name: str, value: float, min_value: float, max_value: float, step: float = 0.1):
        """添加实验参数"""
        self.parameters[name] = {
            'value': value,
            'min': min_value,
            'max': max_value,
            'step': step
        }
        print(f"添加参数: {name} = {value} (范围: {min_value} ~ {max_value})")

    def add_system(self, system: ParticleSystem, name: str = ""):
        """添加粒子系统"""
        self.systems.append({
            'system': system,
            'name': name or f"系统 {len(self.systems) + 1}"
        })
        print(f"添加系统: {name}")

    def initial_condition(self, func: Callable):
        """装饰器：设置初始条件"""
        self.setup_function = func
        return func

    def run(self):
        """运行实验"""
        if not self.systems:
            print("错误：没有添加任何粒子系统")
            return

        print("\n实验开始！")
        print("控制说明:")
        print("  左右方向键: 切换系统")
        print("  +/-键: 调整当前参数")
        print("  空格键: 暂停/继续")
        print("  R键: 重置当前系统")

        running = True
        paused = False

        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print("实验已" + ("暂停" if paused else "继续"))

                    elif event.key == pygame.K_r:
                        self.systems[self.current_system]['system'].reset_system()
                        print("当前系统已重置")

                    elif event.key == pygame.K_LEFT:
                        self.current_system = (self.current_system - 1) % len(self.systems)
                        print(f"切换到系统: {self.systems[self.current_system]['name']}")

                    elif event.key == pygame.K_RIGHT:
                        self.current_system = (self.current_system + 1) % len(self.systems)
                        print(f"切换到系统: {self.systems[self.current_system]['name']}")

                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.adjust_parameter(1)

                    elif event.key == pygame.K_MINUS:
                        self.adjust_parameter(-1)

            # 清屏
            self.screen.fill((20, 20, 30))

            # 获取当前系统
            current = self.systems[self.current_system]
            system = current['system']

            # 更新系统（如果不暂停）
            if not paused:
                system.update()

            # 渲染系统
            stats = system.statistics.get_data()
            system.renderer.render(self.screen, system.particles,
                                   system.force_fields, system.constraints, stats)

            # 绘制实验信息
            self.draw_experiment_info(current['name'])

            # 更新显示
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        print("实验结束")

    def adjust_parameter(self, direction: int):
        """调整参数"""
        if not self.parameters:
            return

        # 获取第一个参数进行调整
        param_name = list(self.parameters.keys())[0]
        param = self.parameters[param_name]

        new_value = param['value'] + direction * param['step']
        new_value = max(param['min'], min(param['max'], new_value))

        if new_value != param['value']:
            param['value'] = new_value
            print(f"参数 {param_name} 调整为: {new_value}")

    def draw_experiment_info(self, system_name: str):
        """绘制实验信息"""
        # 绘制标题
        title_text = self.font.render(f"实验: {self.title}", True, (255, 255, 255))
        self.screen.blit(title_text, (10, 10))

        # 绘制当前系统
        system_text = self.font.render(f"当前系统: {system_name}", True, (200, 200, 255))
        self.screen.blit(system_text, (10, 50))

        # 绘制参数
        y_offset = 90
        for name, param in self.parameters.items():
            param_text = self.font.render(f"{name}: {param['value']:.2f}", True, (200, 255, 200))
            self.screen.blit(param_text, (10, y_offset))
            y_offset += 30

        # 绘制控制说明
        controls = [
            "控制说明:",
            "左右方向键: 切换系统",
            "+/-键: 调整参数",
            "空格键: 暂停/继续",
            "R键: 重置系统"
        ]

        y_offset = self.height - len(controls) * 30 - 20
        for i, line in enumerate(controls):
            control_text = self.font.render(line, True, (255, 255, 200))
            self.screen.blit(control_text, (10, y_offset + i * 30))


# ==================== 使用示例 ====================

def demo_basic_simulation():
    """基本粒子模拟演示"""
    print("=== 基本粒子模拟演示 ===")

    # 创建粒子系统
    system = ParticleSystem(
        width=800,
        height=600,
        num_particles=150,
        gravity=(0, 0.3),
        enable_boundary=True,
        enable_particles_collision=True,
        enable_attraction=True,
        attraction_strength=0.005
    )

    # 添加漩涡场
    vortex = VortexForceField((400, 300), strength=0.3, radius=200)
    system.add_force_field(vortex)

    # 运行系统
    system.run()


def demo_fluid_simulation():
    """流体模拟演示"""
    print("=== 流体模拟演示 ===")

    fluid = FluidSimulation(
        width=800,
        height=600,
        num_particles=800,
        density=1.0,
        viscosity=0.05,
        surface_tension=0.1
    )

    # 添加流体源
    emitter = ParticleEmitter(
        position=(400, 100),
        emission_rate=20,
        velocity_range=(0, 3),
        angle_range=(math.pi / 2 - 0.5, math.pi / 2 + 0.5),
        radius_range=(3, 4),
        color=Color(100, 150, 255),
        lifetime=10.0
    )
    fluid.add_emitter(emitter)

    # 运行流体模拟
    fluid.run()


def demo_plasma_simulation():
    """等离子体模拟演示"""
    print("=== 等离子体模拟演示 ===")

    plasma = PlasmaSimulation(
        width=800,
        height=600,
        electrons=150,
        ions=80,
        magnetic_field=(0, 0, 0.2),
        electric_field=(0.2, 0, 0)
    )

    # 添加电极
    plasma.add_electrode((200, 300), voltage=100, radius=30)
    plasma.add_electrode((600, 300), voltage=-100, radius=30)

    # 运行等离子体模拟
    plasma.run()


def demo_experiment():
    """实验演示"""
    print("=== 物理实验演示 ===")

    # 创建实验
    experiment = Experiment(
        title="气体扩散与布朗运动",
        description="观察不同温度下气体粒子的扩散行为",
        width=1200,
        height=800
    )

    # 添加参数
    experiment.add_parameter('temperature', 300, 100, 500, 10)
    experiment.add_parameter('gravity', 0.0, -1.0, 1.0, 0.1)

    # 创建系统1：高温气体
    system1 = ParticleSystem(width=600, height=800, num_particles=200)
    system1.gravity = Vector2(0, 0)

    # 设置粒子初始位置（集中在左侧）
    for i, particle in enumerate(system1.particles):
        particle.position.x = 100 + (i % 20) * 15
        particle.position.y = 100 + (i // 20) * 15
        particle.velocity = Vector2.random(0, 5)  # 高温：高速
        particle.color = Color(255, 100, 100)  # 红色表示高温

    # 创建系统2：低温气体
    system2 = ParticleSystem(width=600, height=800, num_particles=200)
    system2.gravity = Vector2(0, 0)

    # 设置粒子初始位置（集中在左侧）
    for i, particle in enumerate(system2.particles):
        particle.position.x = 100 + (i % 20) * 15
        particle.position.y = 100 + (i // 20) * 15
        particle.velocity = Vector2.random(0, 2)  # 低温：低速
        particle.color = Color(100, 100, 255)  # 蓝色表示低温

    # 添加系统到实验
    experiment.add_system(system1, "高温气体 (500K)")
    experiment.add_system(system2, "低温气体 (100K)")

    # 运行实验
    experiment.run()


def demo_brownian_motion():
    """布朗运动演示"""
    print("=== 布朗运动演示 ===")

    system = ParticleSystem(
        width=800,
        height=600,
        num_particles=5,  # 少量大粒子
        gravity=(0, 0),
        enable_boundary=True,
        enable_particles_collision=True
    )

    # 创建大粒子（布朗粒子）
    for particle in system.particles:
        particle.radius = 10
        particle.mass = 10.0
        particle.color = Color(255, 200, 100)
        particle.fixed = False

    # 添加大量小粒子（溶剂分子）
    for _ in range(300):
        x = random.uniform(50, system.width - 50)
        y = random.uniform(50, system.height - 50)

        small_particle = Particle(
            position=(x, y),
            velocity=(random.uniform(-5, 5), random.uniform(-5, 5)),
            radius=2,
            mass=0.1,
            color=Color(100, 150, 255, 150),
            lifetime=-1
        )
        system.particles.append(small_particle)

    # 设置渲染模式
    system.renderer.show_trails = True
    system.renderer.trail_length = 100

    # 运行布朗运动模拟
    system.run()


def demo_custom_simulation():
    """自定义模拟演示"""
    print("=== 自定义模拟演示 ===")

    # 创建系统
    system = ParticleSystem(
        width=1000,
        height=800,
        num_particles=0,  # 开始时不创建粒子
        gravity=(0, 0.2),
        enable_boundary=True,
        enable_particles_collision=True
    )

    # 添加各种力场
    system.add_gravity_well((300, 300), strength=200, radius=150)
    system.add_gravity_well((700, 300), strength=200, radius=150)

    vortex = VortexForceField((500, 400), strength=0.4, radius=250)
    system.add_force_field(vortex)

    # 添加约束
    circle_constraint = CircleConstraint((500, 400), 350, elasticity=0.9)
    system.add_constraint(circle_constraint)

    # 添加发射器
    emitter = ParticleEmitter(
        position=(500, 200),
        emission_rate=15,
        velocity_range=(-1, 1),
        angle_range=(math.pi / 2 - 0.3, math.pi / 2 + 0.3),
        radius_range=(2, 4),
        color=Color.random(),
        burst_mode=False
    )
    system.add_emitter(emitter)

    # 设置渲染
    system.renderer.set_render_mode(RenderMode.TRAILS)
    system.renderer.show_vectors = True

    print("自定义模拟已创建")
    print("包含：2个重力井，1个漩涡场，1个圆形约束，1个粒子发射器")

    # 运行模拟
    system.run()


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    print("=" * 50)
    print("粒子模拟系统 - Particle Simulation System")
    print("版本 2.0.0")
    print("=" * 50)
    print("\n请选择演示模式：")
    print("1. 基本粒子模拟")
    print("2. 流体模拟")
    print("3. 等离子体模拟")
    print("4. 布朗运动演示")
    print("5. 物理实验演示")
    print("6. 自定义模拟")
    print("0. 退出")

    choice = input("\n请输入选择 (0-6): ").strip()

    if choice == "1":
        demo_basic_simulation()
    elif choice == "2":
        demo_fluid_simulation()
    elif choice == "3":
        demo_plasma_simulation()
    elif choice == "4":
        demo_brownian_motion()
    elif choice == "5":
        demo_experiment()
    elif choice == "6":
        demo_custom_simulation()
    elif choice == "0":
        print("再见！")
    else:
        print("无效选择，运行基本演示...")
        demo_basic_simulation()


if __name__ == "__main__":
    # 直接运行主函数
    main()
from particle_system import ParticleSystem

# 创建系统
system = ParticleSystem(width=800, height=600, num_particles=100)

# 添加力场
system.add_gravity_well((400, 300), strength=100, radius=200)

# 运行模拟
system.run()