
"""
COSMAI Framework - Quantum Field Theory Module
File: cosmai/detectors/detector_base.py

Agente D: Desarrollo Python (Versión Refinada v1.1)
Descripción: Interfaz abstracta pura, validaciones físicas y tipos base 
para detectores cuánticos en QFT.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional, Protocol, Union
import math
import numpy as np


# =====================================================================
# Objetos de Configuración y Tipos Tipados (Evita **kwargs: Any)
# =====================================================================

class IntegrationMethod(Enum):
    """Métodos numéricos soportados por el calculador de respuesta."""
    QUAD = auto()
    GAUSS_KRONROD = auto()
    MONTE_CARLO = auto()
    GPU_ACCELERATED = auto()


@dataclass(frozen=True)
class ResponseOptions:
    """
    Objeto de configuración inmutable para parametrizar las operaciones
    de cálculo numérico de la respuesta sin perder tipado estático.
    """
    method: IntegrationMethod = IntegrationMethod.QUAD
    rtol: float = 1e-8
    atol: float = 1e-10
    limit: int = 200
    subsample_grid: int = 1000


# =====================================================================
# Protocólos e Interfaces Contrato para Componentes Auxiliares
# =====================================================================

class Trajectory(Protocol):
    """Interfaz abstracta que describe la trayectoria del detector x(τ)."""

    def __call__(self, tau: float) -> Union[Tuple[float, float, float, float], np.ndarray]:
        """
        Devuelve las coordenadas espaciotemporales (t, x, y, z) para un tiempo propio τ.
        """
        ...


class SwitchingFunction(Protocol):
    """Interfaz abstracta para la función de encendido/corte χ(τ)."""

    def __call__(self, tau: float) -> float:
        """Devuelve el valor de conmutación (entre 0 y 1) en el tiempo propio τ."""
        ...


class CorrelationProvider(Protocol):
    """
    Interfaz abstracta genérica para proveedores de funciones de correlación
    de dos puntos <Φ(x1)Φ(x2)> (Wightman, Hadamard, Green, etc.).
    """

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> complex:
        """Calcula el valor del correlador de 2 puntos en dos eventos espaciotemporales."""
        ...


# Alias de retrocompatibilidad / claridad semántica
WightmanProvider = CorrelationProvider


# =====================================================================
# Estado Interno y Espacio de Hilbert del Detector
# =====================================================================

@dataclass
class InternalState:
    """Representa el estado interno o densidad del detector en el espacio de Hilbert."""
    label: str = "|0>"
    probabilities: Tuple[float, ...] = (1.0, 0.0)  # P(ground), P(excited)


# =====================================================================
# Clase Base Abstracta para Detectores Cuánticos
# =====================================================================

class AbstractDetector(ABC):
    """
    Clase Base Abstracta para todos los modelos de detectores cuánticos en COSMAI.

    Representa la abstracción de un detector que interactúa con un campo cuántico.
    Define los parámetros físicos, validaciones y contrato funcional sin ejecutar 
    integración numérica directa.
    """

    def __init__(
        self,
        energy_gap: float,
        coupling: float,
        trajectory: Trajectory,
        switching: SwitchingFunction,
        initial_state: Optional[InternalState] = None,
    ) -> None:
        """
        Inicializa y valida los parámetros físicos universales del detector.

        Args:
            energy_gap (float): Gap energético Ω del detector (Ω > 0 para absorción).
            coupling (float): Constante de acoplamiento perturbativo λ con el campo.
            trajectory (Trajectory): Curva parametrizada por el tiempo propio x(τ).
            switching (SwitchingFunction): Perfil temporal de activación χ(τ).
            initial_state (Optional[InternalState]): Estado cuántico inicial.

        Raises:
            ValueError: Si los parámetros físicos son finitos no válidos (NaN, Inf).
        """
        self._validate_physical_params(energy_gap, coupling)

        self._energy_gap: float = float(energy_gap)
        self._coupling: float = float(coupling)
        self._trajectory: Trajectory = trajectory
        self._switching: SwitchingFunction = switching
        self._state: InternalState = initial_state or InternalState()

    @staticmethod
    def _validate_physical_params(energy_gap: float, coupling: float) -> None:
        """Valida la integridad física y numérica de los parámetros de entrada."""
        if not math.isfinite(energy_gap):
            raise ValueError(f"El gap energético 'energy_gap' debe ser finito. Recibido: {energy_gap}")
        if not math.isfinite(coupling):
            raise ValueError(f"La constante de acoplamiento 'coupling' debe ser finita. Recibido: {coupling}")
        if coupling < 0.0:
            raise ValueError(f"La constante de acoplamiento 'coupling' (λ) debe ser no negativa. Recibida: {coupling}")

    # -----------------------------------------------------------------
    # Propiedades Físicas (Inmutables por defecto)
    # -----------------------------------------------------------------

    @property
    def energy_gap(self) -> float:
        """Retorna el gap de energía Ω del detector."""
        return self._energy_gap

    @property
    def coupling(self) -> float:
        """Retorna la constante de acoplamiento λ (perturbativa)."""
        return self._coupling

    @property
    def trajectory(self) -> Trajectory:
        """Retorna el objeto que define la trayectoria x(τ)."""
        return self._trajectory

    @property
    def switching(self) -> SwitchingFunction:
        """Retorna la función de conmutación χ(τ)."""
        return self._switching

    @property
    def state(self) -> InternalState:
        """Retorna el estado cuántico interno actual del detector."""
        return self._state

    # -----------------------------------------------------------------
    # Métodos Abstractos (Contrato Físico y Computacional)
    # -----------------------------------------------------------------

    @abstractmethod
    def response_function(
        self,
        correlator: CorrelationProvider,
        options: Optional[ResponseOptions] = None
    ) -> float:
        """
        Calcula la función de respuesta F(Ω) del detector.

        Note:
            F(Ω) depende únicamente de la trayectoria, la función de corte
            y el correlador de 2 puntos. La constante de acoplamiento λ² NO 
            está incluida en F(Ω).
        """
        pass

    @abstractmethod
    def excitation_probability(
        self,
        correlator: CorrelationProvider,
        options: Optional[ResponseOptions] = None
    ) -> float:
        """
        Calcula la probabilidad de excitación P_e = λ² F(Ω) + O(λ⁴) a primer
        orden en teoría de perturbaciones.
        """
        pass

    @abstractmethod
    def transition_rate(
        self,
        tau: float,
        correlator: CorrelationProvider,
        options: Optional[ResponseOptions] = None
    ) -> float:
        """
        Calcula la tasa de transición instantánea dP_e/dτ en un instante de 
        tiempo propio τ específico.
        """
        pass

    @abstractmethod
    def sample_events(
        self,
        n_samples: int,
        correlator: CorrelationProvider,
        options: Optional[ResponseOptions] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Genera realizaciones estocásticas Monte Carlo de eventos de detección.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Restablece el estado interno del detector a su configuración inicial |0>."""
        self._state = InternalState()
