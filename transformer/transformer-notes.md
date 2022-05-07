Transformer
===========

## Self-attention

The input is $t$ $k$-dim vectors, $x_t$.
Queries, keys and values are computed with separate $k$-by-$k$ matrices:

$q_t={W^{(q)}}\cdot x_t$,

$k_t={W^{(k)}}\cdot x_t$,

$v_t={W^{(v)}}\cdot x_t$.

With batch, head and vector dimension, it becomes for example

$q_{bthk} = {W^{(q)}}_{hke}x_{bte}$.

The weights become

$w_{bhtu} = q_{bthk} k_{buhk}$.

Then the output is

$y'_{bthk} = w_{bhtu} v_{buhk}$.

To unify the heads, one computes

$y_{btk} =  y'_{bthl}W^{(u)}_{lhe}$.

