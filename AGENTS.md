# General Code Conventions

## 1. Do not write defensive code

Write simple, direct code. Do not add defensive logic for internal code paths unless it is required by a real external boundary.

External boundaries include:
- user input
- file I/O
- network calls
- subprocesses
- third-party APIs
- hardware/robot interfaces
- serialization/deserialization

Avoid:
- broad `try/except` blocks
- silent fallbacks
- redundant `None` checks
- speculative edge-case handling
- default values that hide bugs
- validation for impossible internal states
- code paths that continue after an invariant has been violated
- `dict.get(key, default)` or `cfg.get(key, default)` for required values; use `dict["key"]` or `cfg["key"]`

Prefer fail-fast behavior. If an internal invariant is broken, let the error surface clearly.

Good:

```python
def compute_loss(pred, target):
    return ((pred - target) ** 2).mean()
```

Avoid:

```python
def compute_loss(pred, target):
    if pred is None or target is None:
        return 0.0

    try:
        return ((pred - target) ** 2).mean()
    except Exception:
        return 0.0
```

## 2. Prefer functional programming style

Prefer pure functions and explicit data flow.

Follow these rules:
- Functions should take all required inputs explicitly.
- Functions should return new values instead of mutating inputs in place.
- Avoid global mutable state.
- Avoid hidden side effects.
- Keep computation separate from I/O, logging, plotting, saving files, robot execution, or environment interaction.
- Prefer small composable functions over large stateful classes.
- Use classes only when they clearly represent long-lived state, external resources, or interfaces.

Good:

```python
def normalize_keypoints(keypoints, image_size):
    width, height = image_size
    scale = np.array([width, height])
    return keypoints / scale
```

Avoid:

```python
def normalize_keypoints(self):
    self.keypoints = self.keypoints / self.image_size
```

## 3. Do not overwrite or mutate `cfg`

Treat all config objects as immutable.

Never mutate `cfg` inside a function.

Avoid:

```python
def train(cfg):
    cfg.training.batch_size = 64
    cfg.model.hidden_dim = 256
```

Prefer deriving local variables from `cfg`:

```python
def train(cfg):
    batch_size = cfg.training.batch_size
    hidden_dim = cfg.model.hidden_dim
```

If a modified config is truly needed, create a new independent config object instead of overwriting the existing one.

## 4. Config ownership rule

Do not pass overlapping config objects at the same time.

If one config is a subset of another config, use only the parent config.

Good:

```python
train(cfg)
```

Avoid:

```python
train(cfg, cfg.training)
```

because `cfg.training` is already contained inside `cfg`.

Only pass multiple config objects when they are independent and non-overlapping.

Good:

```python
run_training(model_cfg, data_cfg, optim_cfg)
```

This is acceptable only if `model_cfg`, `data_cfg`, and `optim_cfg` are independent configuration domains.

Avoid:

```python
run_training(cfg, model_cfg)
```

if `model_cfg` is already contained inside `cfg`.

The rule is:

- If `child_cfg` is contained in `parent_cfg`, use only `parent_cfg`.
- If two configs describe independent domains, they may be used together.
- Never mutate `cfg`.
- Never overwrite config values to control local behavior.
- Prefer deriving local variables from `cfg` instead of modifying `cfg`.

## 5. Keep code simple

Prefer the simplest implementation that matches the requested behavior.

Do not introduce unnecessary abstractions, registries, factories, wrappers, inheritance, or compatibility layers unless they are clearly needed by the existing codebase.

Prefer:
- clear names
- explicit arguments
- straightforward control flow
- small functions
- minimal side effects

Do not extract tiny helpers by default. If a function is shorter than 5 lines
and is not reused at least 3 times, inline the logic at the call site instead
of defining a new function. Exceptions are allowed when the helper gives a
domain concept a necessary name, isolates an external boundary, or prevents a
real readability problem.

Assume normal project code is run from the project root. Do not modify
`sys.path` to make imports work, and do not append or insert project paths at
runtime. Path injection is allowed only for temporary analysis scripts, debug
snippets, or one-off local experiments that are not part of the main codebase.
