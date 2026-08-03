![GitHub Downloads (all assets, latest release)](https://img.shields.io/github/downloads/nicholasconnors/erebus/latest/total)
![GitHub Pages Deployment](https://img.shields.io/github/actions/workflow/status/nicholasconnors/erebus/publish_docs.yaml?label=pages)

Erebus is a rocky-exoplanet secondary eclipse aperture-photometry lightcurve fitting pipeline made for use with the Mid Infra-Red Instrument aboard the James Webb Space Telescope, starting with `calints.fits` files available on the [Barbara A. Mikulski Archive for Space Telescopes](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html). Optionally Erebus can be run starting with `uncal.fits` files, in which case it calls [Eureka!](https://github.com/kevin218/Eureka) to perform the first two stages of the pipeline. If you use this functionality, be sure to properly [cite Eureka](https://eurekadocs.readthedocs.io/en/latest/index.html#citing-eureka-and-its-dependencies).

For issue reporting or feedback suggestions click [here](https://github.com/nicholasconnors/erebus/issues).

For the setup used when writing Connors et al. (2025) click [here](https://github.com/nicholasconnors/erebus/tree/main/connors_et_al_2025). For the setup used when writing Connors et al. (2026) for GJ 3929 b click [here](https://github.com/nicholasconnors/erebus/tree/main/connors_et_al_2026).

If you use Erebus in a scientific publication we will ask you cite our paper:

```
@ARTICLE{2025ApJ...989L..11C,
       author = {{Connors}, Nicholas J. and {Monaghan}, Christopher and {Benneke}, Bj{\"o}rn and {Dang}, Lisa},
        title = "{Uniform Reanalysis of JWST MIRI 15 {\ensuremath{\mu}}m Exoplanet Eclipse Observations Using Frame-normalized Principal Component Analysis}",
      journal = {\apjl},
     keywords = {Exoplanets, Exoplanet atmospheres, Planetary atmospheres, 498, 487, 1244, Earth and Planetary Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = aug,
       volume = {989},
       number = {1},
          eid = {L11},
        pages = {L11},
          doi = {10.3847/2041-8213/adee0d},
archivePrefix = {arXiv},
       eprint = {2507.02052},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...989L..11C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

For other uses of Erebus you can cite:

```
@ARTICLE{2026ApJ..1006L...6C,
       author = {{Connors}, Nicholas J. and {Monaghan}, Christopher and {Benneke}, Bj{\"o}rn and {Dang}, Lisa and {Roy}, Pierre-Alexis},
        title = "{GJ 3929 b as the First Complete Rocky Worlds DDT Data Set}",
      journal = {\apjl},
     keywords = {Exoplanets, Exoplanet atmospheres, Planetary atmospheres, 498, 487, 1244, Earth and Planetary Astrophysics},
         year = 2026,
        month = jul,
       volume = {1006},
       number = {1},
          eid = {L6},
        pages = {L6},
          doi = {10.3847/2041-8213/ae8018},
archivePrefix = {arXiv},
       eprint = {2606.07511},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026ApJ..1006L...6C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
