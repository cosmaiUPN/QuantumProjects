/*
COSMAI - Quantum Vacuum Detector Prototype
Versión conceptual 

```
Idea:
- |0> = detector en estado fundamental
- |1> = detector excitado
- El vacío cuántico se representa mediante una probabilidad
  efectiva de excitación calculada externamente.
- La probabilidad se transforma en una rotación Ry.

Futuras extensiones:
- Correladores de Wightman
- Detectores Unruh-DeWitt
- Entanglement Harvesting
- Conjuntos causales discretizados
- Simulación de campos en lattice
```

*/

namespace COSMAI.QuantumVacuum {

```
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Measurement;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Convert;

// Convierte una probabilidad física en un ángulo
// de rotación para el detector.
function ProbabilityToAngle(p : Double) : Double {
    return 2.0 * ArcSin(Sqrt(p));
}

// Detector elemental.
operation VacuumDetector(
    excitationProbability : Double
) : Result {

    use detector = Qubit();

    let theta = ProbabilityToAngle(
        excitationProbability
    );

    Ry(theta, detector);

    let result = M(detector);

    Reset(detector);

    return result;
}

// Ejecuta múltiples mediciones.
operation SampleVacuumResponse(
    excitationProbability : Double,
    shots : Int
) : Double {

    mutable excitations = 0;

    for _ in 1..shots {

        let measurement =
            VacuumDetector(
                excitationProbability
            );

        if measurement == One {
            set excitations += 1;
        }
    }

    return IntAsDouble(excitations)
         / IntAsDouble(shots);
}

@EntryPoint()
operation Main() : Unit {

    // Valor ficticio.
    // En una simulación real provendría
    // de correladores de Wightman.

    let pVacuum = 0.08;

    let response =
        SampleVacuumResponse(
            pVacuum,
            10000
        );

    Message(
        $"Vacuum excitation probability = {pVacuum}"
    );

    Message(
        $"Measured detector response = {response}"
    );
}
```

}
