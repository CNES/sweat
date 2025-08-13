# Algorithm

EVASPA is an ensemble approach which uses a variety of contextual algorithms based on the evaporative fraction models. 
The average and dispersion of the results of these models are then computed to get the final EF fraction, and an idea of its reliability.

## Evaporative fraction model

Evaporative fraction models consider the calculation of latent heat flux, $LE$, as

$$
LE = EF \times A
$$

where $EF$ is the evaporative fraction (-) and A is the available energy ($W.m^{-2}$). 
The evaporative fraction expresses the quantity of energy which is allocated to evapotranspiration in regards to the available energy. 
Depending on the time scale, the available energy $A$ is defined as $R_n$ or $R_n – G$. 
On a daily scale, $G$ is considered to compensate between daytime and night-time values. 
So, we only consider $R_n$ for a daily calculation and Rn-G for the instantaneous estimation.
$EF$ is defined as 

$$
EF = \frac{LE}{LE+H} = \frac{LE}{R_n-G}
$$

To compute $EF$, EVASPA is based on the model called Simplified Surface Energy Balance Index (S-SEBI) that has been developed to solve 
the surface energy balance with remote sensing techniques on a pixel-by-pixel basis (Roering 2000). 
It has been observed that the surface temperature and reflectance of areas subject to constant atmospheric forcing are correlated, 
and that the relationships can be applied to determine the effective properties of the land surface (Menenti 1989; Bastiaanssen 1995).
Up to a certain temperature, surface temperature can be described as “evaporation-controlled”, because the change in temperature is 
the result of reduced evaporation due to reduced availability of soil moisture. In this case, the increase in excess sensible heat flux exceeds 
the decrease in net radiation due to increased reflectance. Above a certain reflectance threshold, surface temperature decreases as reflectance increases. 
This is because soil moisture has decreased to such an extent that no evaporation can take place in this case. The available energy is 
therefore used solely to heat the surface. However, due to the increase in reflectance, the available energy decreases as net radiation decreases 
(more is reflected). This process leads to a decrease in temperature as the reflectance increases. Temperature is then said to be “radiation-controlled”. 
A schematic representation of the S-SEBI is shown in figure.

![EF_schema](images/ef_schema.png){ width="400" }
/// caption
Schematic representation of the relationship between surface reflectance and land surface temperature together the principles of S-SEBI
///

In S-SEBI article, Roerink studies the relation between surface albedo (issued from surface reflectances), surface temperature and evaporative fraction. 
If the dry curve, i.e. "radiation-controlled" curve and the wet curve, i.e. "evaporation-controlled" curve can be determined, S-SEBI calculates the evaporative 
fraction as follows: for each pixel with the surface albedo, , and surface temperature, the hypothetical values of land surface temperature are determined:

* For fully wet conditions, i.e. Twet where LE = Rn-G and H= 0 
* For fully dry conditions, i.e. Tdry where H = Rn - G and LE=0

$$
EF = \frac{T_{dry}-T}{T_{dry}-T{wet}}
$$

## Evapotranspiration from evaporative fraction

From evaporative fraction $EF$, latent heat flux can be expressed as 

$$
LE = EF\times(R_n-G) 
$$

The net radiation $R_n$ ($W.m^{-2}$) corresponds to the radiation balance at the surface:

$$
R_n = R_{SD} \times (1-\alpha) + \epsilon \times(R_{LD}-\sigma T^{4})
$$ 

where:

* $R_{SD} ($W. m^{-2}$) is the downwelling shortwave radiation, 
* $\alpha$ the albedo (-), 
* $\epsilon$ the broad band surface emissivity (-), 
* $R_{LD}$ the downwelling longwave radiation (or atmospheric radiation ($W.m^{-2}$)), 
* $\sigma$ the Stefan-Boltzmann constant and 
* $T$ the radiative temperature of the surface (K). 

The ground heat flux $G$ depends theoretically on the temperature gradient within the top centimeters of the soil and on the thermal properties (capacity, conductivity) of the soil surface layer.
$G$ can also be expressed as a fraction $\xi$ of Rn being related to the amount of vegetation or the vegetation fraction cover (see Kustas et al. 1993, Kpemlie 2009):

$$
\xi = \frac{G}{R_n} 
$$
$\xi$ can be modeled using equations based on vegetation indices, see Kustas et al. 1993.

Eventually, instantaneous evapotranspiration is obtained from latent heat flux:
$$
ET = \frac{LE}{L} = \frac{EF(R_n-G)}{L}
$$
Where L is the latent heat of vaporization.

## Extrapolation to daily evapotranspiration
On days with a TRISHNA acquisition, the construction of daily ET is done through an extrapolation (or upscaling) of instantaneous data to the daily scale thanks to the use of the solar radiation ratio. 
As solar radiation is also a simple factor to use, upscaling of instantaneous evapotranspiration  is done here through the ratio of solar radiation at the time of TRISHNA acquisition (obtained as solar radiation TRISHNA product) and the daily solar radiation so that:

$$
ET^{daily} = \frac{R_{SD}^{daily}}{R_{SD}}ET
$$
where $ET^{daily}$ ($mm.d^{-1}$) is the daily evapotranspiration, $R_{SD}$ is the downwelling shortwave radiation at the time of TRISHNA acquisition, $R_{SD}^{daily}$ is the daily value of downwelling shortwave radiation.