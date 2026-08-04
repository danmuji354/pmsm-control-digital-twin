# PMSM model

The state is `[id, iq, mechanical_speed, electrical_angle]`. Salient dq inductances, permanent-magnet flux, copper resistance, inertia, viscous friction, pole pairs, load torque, and a circular phase-voltage bound are explicit parameters.

The predictive controller inverts one Euler prediction step, then projects its voltage request onto the feasible disk. The FOC baseline uses PI current loops plus cross-coupling feedforward.

