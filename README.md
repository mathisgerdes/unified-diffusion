# Generation with Unified Diffusion

Diffusion utilities for consistent basis transforms, schedules, and sampling.
Built around whitened and unwhitened spaces with data, score and epsilon conversions.

This code implements the methods described in the paper: [GUD: Generation with Unified Diffusion](https://arxiv.org/abs/2410.02667)

## Tutorial Dependencies

The tutorial notebooks require additional packages beyond the core dependencies.
Install them with:

```bash
pip install optax matplotlib torchvision
```

Or individually:
- `optax` - Gradient processing and optimization library for JAX
- `matplotlib` - Plotting library
- `torchvision` - Computer vision datasets and transforms
- `ipython` - Interactive Python shell (for notebook display utilities)
