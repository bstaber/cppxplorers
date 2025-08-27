# Optimizers

This chapter documents the small optimization module used in the project: a minimal runtime‑polymorphic interface `Optimizer` with two concrete implementations, Gradient Descent and Momentum. It is designed for clarity and easy swapping of algorithms in training loops.


## Problem setting

Given parameters $\mathbf{w}\in\mathbb{R}^d$ and a loss $\mathcal{L}(\mathbf{w})$, an optimizer updates weights using the gradient
$$
\mathbf{g}_t=\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_t).
$$
Each algorithm defines an update rule $\mathbf{w}_{t+1} = \Phi(\mathbf{w}_t,\mathbf{g}_t,\theta)$ with hyper‑parameters $\theta$ (e.g., learning rate, momentum).


## API overview

<details>
<summary>Click here to view the full implementation: <b>include/cppx/opt/optimizers.hpp</b>. We break into down in the sequel of this section. </summary>

```cpp
{{#include ../../crates/simple_optimizers/include/optimizers.hpp}}
```
</details>

Design choices
- A small virtual interface to enable swapping algorithms at runtime.
- `std::unique_ptr<Optimizer>` for owning polymorphism; borrowing functions accept `Optimizer&`.
- Exceptions (`std::invalid_argument`) signal size mismatches.


## Gradient descent

Update rule
$$
\mathbf{w}_{t+1}=\mathbf{w}_{t}-\eta\,\mathbf{g}_t ,
$$
with learning rate $\eta>0$.

Implementation
```cpp
void GradientDescent::step(std::vector<double>& w,
                           const std::vector<double>& g) {
  if (w.size() != g.size()) throw std::invalid_argument("size mismatch");
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] -= lr_ * g[i];
  }
}
```

## Momentum-based gradient descent

Update rule
$$
\begin{aligned}
\mathbf{v}_{t+1} &= \mu\,\mathbf{v}_{t} + \eta\,\mathbf{g}_t, \\\\
\mathbf{w}_{t+1} &= \mathbf{w}_{t} - \mathbf{v}_{t+1},
\end{aligned}
$$
with momentum $\mu\in[0,1)$ and learning rate $\eta>0$.

Implementation
```cpp
Momentum::Momentum(double learning_rate, double momentum, std::size_t dim)
  : lr_(learning_rate), mu_(momentum), v_(dim, 0.0) {}

void Momentum::step(std::vector<double>& w, const std::vector<double>& g) {
  if (w.size() != g.size()) throw std::invalid_argument("size mismatch");
  if (v_.size() != w.size()) throw std::invalid_argument("velocity size mismatch");

  for (std::size_t i = 0; i < w.size(); ++i) {
    v_[i] = mu_ * v_[i] + lr_ * g[i];
    w[i] -= v_[i];
  }
}
```

## Using the optimizers

### Owning an optimizer (runtime polymorphism)

```cpp
#include <memory>
#include "cppx/opt/optimizers.hpp"

using namespace cppx::opt;

std::vector<double> w(d, 0.0), g(d, 0.0);

// Choose an algorithm at runtime:
std::unique_ptr<Optimizer> opt =
    std::make_unique<Momentum>(/*lr=*/0.1, /*mu=*/0.9, /*dim=*/w.size());

for (int epoch = 0; epoch < 100; ++epoch) {
  // ... compute gradients into g ...
  opt->step(w, g);           // updates w in place
}
```

### Borrowing an optimizer (no ownership transfer)

```cpp
void train_one_epoch(Optimizer& opt,
                     std::vector<double>& w,
                     std::vector<double>& g) {
  // ... fill g ...
  opt.step(w, g);
}
```

### API variations (optional)

If C++20 is available, `std::span` can make the interface container‑agnostic:

```cpp
// virtual void step(std::span<double> w, std::span<const double> g) = 0;
```
