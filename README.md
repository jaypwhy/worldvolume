# worldvolume

An open-source Python toolkit for solving field equations on D-brane worldvolumes
with mixed Neumann / Dirichlet / Robin boundary conditions.

Status: in development. The first working version solves the Klein-Gordon equation
on a rectangular worldvolume with an independently chosen boundary condition per
direction, and validates the computed spectrum against closed-form analytic results.

See `notes/bc-conventions.md` for the derivation and conventions this implementation
is built against.
