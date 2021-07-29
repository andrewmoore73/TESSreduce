[![Build and Test [Python 3.7, 3.8, 3.9]](https://github.com/CheerfulUser/TESSreduce/actions/workflows/build.yml/badge.svg)](https://github.com/CheerfulUser/TESSreduce/actions/workflows/build.yml)

![plot](./figs/header.png)

With this package that builds on lightkurve, you can reduce TESS data while preserving transient signals. You can supply a TPF or give coordinates and sector to construct a TPF with TESScut. The background subtraction accounts for the smooth background and
detector straps. Alongisde background subtraction TESSreduce also aligns images, performs difference imaging, and can even detect transient events! 

An additional component that is in development is calibration of TESS photometry, and reliably link muti-sector light curves.

TESSreduce can be installed through pip:

`pip install git+https://github.com/CheerfulUser/TESSreduce.git`

Example reduction for SN 2018fub:
```python
import tessreduce as tr
obs = tr.sn_lookup('sn2018fub')
```
```
|   Sector | Covers   |   Time difference  |
|          |          |             (days) |
|----------+----------+--------------------|
|        2 | True     |                  0 |
|       29 | False    |                721 |
```

```python
tess = tr.tessreduce(obs_list=obs,plot=False,reduce=True)
```
![plot](./figs/fub.png)

**OR**
```python
import tessreduce as tr
ra = 10.127
dec = -50.687
sector = 2
tess = tr.tessreduce(ra=ra,dec=dec,sector=sector)
tess.reduce()
# If you want to remove residual background trends from a bright as best as possible
detrend = tess.detrend_transient()
```


# Flux calibration

TESSreduce can calibrate TESS counts to physical flux, or AB magnitudes, by using PS1 data. If your field is dec > -30 and you want a flux calibrated light curve then use:
```python
tess.to_flux()
```
**OR**
```python
tess.to_mag()
```
Several options are available for flux and are interchangeable, however, mag is currently not reversible. To easily plot the resulting light curve:
```python
tess.plotter()
```
![plot](./figs/fub_cal.png)


There are a lot of other functions burried in TESSreduce which currently aren't well documented, so for more information contact me at: rridden@stsci.edu
