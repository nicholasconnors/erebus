## Results

LHS1140c, LHS1478b, TOI1468b results were made with 0.7.0 of the software (did not properly support eccentricity but these planets were treated as having circular orbits).

TRAPPIST-1b and c FN-PCA were made with 0.7.0 of the software (eccentricity supported but not optimized so could easily get stuck without converging: however these runs all converged.)

TRAPPIST-1b and c exponential fts were made with 0.7.1 of the software (eccentricity convergence bug fixed which had affected these runs previously).

For TRAPPIST-1c separate runs were done allowing or forbidding negative eclipse depths. This is due to the software sometimes mistaking the eclipse egress as a negative eclipse ingress allowed within the timing constraints on the eccentricity. The forced positive eclipse depth was only used when the data was clearly inconsistent with a missed or 0 eclipse depth.

For the TRAPPIST-1c exponential joint fit run the parameters did not all converge (the systematic model parameters did not). The eclipse depth value however did converge and is used in Table 2 of the paper.

Chain plot pdf images aren't included because they are too large for GitHub