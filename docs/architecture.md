# Architecture Overview

The `fnblur` library uses C++17 and platform-specific intrinsics (ARMv8 NEON) to achieve high-performance image processing.

```mermaid
graph TD
    User[Python User] --> API[fnblur.gaussian]
    API --> PyBind[PyBind11 Wrapper]
    PyBind --> Cpp[C++ Implementation]

    subgraph "C++ Core (src/fast_blur.cpp)"
        Cpp --> Check[Input Validation]
        Check --> Separable[Separable Filter Strategy]
        Separable --> Horizontal[Horizontal Pass]
        Separable --> Vertical[Vertical Pass]

        Horizontal --> NEON_H{NEON Intrinsics?}
        Vertical --> NEON_V{NEON Intrinsics?}

        NEON_H -->|Yes| Vec_H[vld1q_u8 / vmulq_f32]
        NEON_H -->|No| Scalar_H[Scalar Fallback]

        NEON_V -->|Yes| Vec_V[vld1q_u8 / vmulq_f32]
        NEON_V -->|No| Scalar_V[Scalar Fallback]
    end

    Vec_H --> Result_H[Intermediate Buffer]
    Scalar_H --> Result_H

    Result_H --> Vertical

    Vec_V --> Output[Output Image]
    Scalar_V --> Output
```

## Key Optimization Strategies

1.  **Separable Filters**: A 2D Gaussian blur is separated into two 1D passes (horizontal then vertical), reducing complexity from O(K^2) to O(K).
2.  **NEON SIMD**: Critical loops are vectorized using ARM NEON intrinsics to process multiple pixels in parallel.
3.  **Direct Buffer Access**: Pybind11 allows direct access to NumPy array buffers, avoiding unnecessary data copying.
