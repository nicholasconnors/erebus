# API - Configuring Erebus

> <p style="font-size: 24px;">Planet</p>
> To find the parameters for a planet we recommend you use the [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) where `t0` is the time of conjunction (`tc` on the Exoplanet archive), `p` is the orbital period in days, `rp_rstar` is the ratio of the planet's radius to the star's radius, `a_rstar` is the ratio of the planet's semi-major axis to the star's radius, `inc` is the inclination of the planet (in degrees), `ecc` is the eccentricity of the planet (0 inclusive to 1 exclusive), and `w` is the agument of periastron of the planet (in degrees).
> ::: utility.planet.Planet
    options:
        show_signature_annotations: false
        show_root_heading: false
        summary: true
        show_if_no_docstring: false

***

> <p style="font-size: 24px;">ErebusRunConfig</p>
> ::: utility.run_cfg.ErebusRunConfig
    options:
        show_signature_annotations: false
        show_root_heading: false
        summary: true
        show_if_no_docstring: false

***

> <p style="font-size: 24px;">Parameters</p>
> ::: utility.bayesian_parameter.Parameter
    options:
        show_signature_annotations: false
        show_root_heading: false
        summary: true
        show_if_no_docstring: false