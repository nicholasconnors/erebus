## Results

LHS1140c, LHS1478b, TOI1468b results were made with 0.7.0 of the software (did not properly support eccentricity but these planets were treated as having circular orbits).

TRAPPIST-1b and c FN-PCA were made with 0.7.0 of the software (eccentricity supported by not optimized so could easily get stuck without converging: however these runs all converged.)

TRAPPIST-1b and c exponential fts were made with 0.7.1 of the software (eccentricity convergence bug fixed which had affected these runs previously).

For TRAPPIST-1c separate runs were done allowing or forbidding negative eclipse depths. This is due to the software sometimes mistaking the eclipse egress as a negative eclipse ingress allowed within the timing constraints on the eccentricity. 