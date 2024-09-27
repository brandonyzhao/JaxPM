# %%
import os
import sys

import argparse 
parser = argparse.ArgumentParser()

parser.add_argument('-d', type=str, default='0')
parser.add_argument('-seed', type=int, default=0)

args = parser.parse_args()

sys.path.append('../')
os.environ['CUDA_VISIBLE_DEVICES'] = args.d
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# %%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import jax_cosmo as jc

from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint, cic_paint_2d
from jaxpm.pm import linear_field, lpt, make_ode_fn
from jaxpm.lensing import density_plane

import jax_cosmo.constants as constants

from jax.scipy.ndimage import map_coordinates
from jaxpm.utils import gaussian_smoothing

import astropy.units as u

import matplotlib.pyplot as plt

# %%
# Below are a few parameters that generate a low resolution version of the k-TNG simulation

box_size = [350.,350.,4000.]    # Transverse comoving size of the simulation volume
nc = [256, 256, 256]              # Number of transverse voxels in the simulation volume
plane_res = 256
lensplane_width = 200         # Width of each lensplane
# lensplane_width = 62.5         # Width of each lensplane
field_size = 5                  # Size of the lensing field in degrees
field_npix = 300                # Number of pixels in the lensing field
z_source = jnp.linspace(0,2)    # Source planes
seed = args.seed                        # Seed for nbody initial conditions

# Defining the coordinate grid for lensing map
xgrid, ygrid = jnp.meshgrid(jnp.linspace(0, field_size, field_npix, endpoint=False), # range of X coordinates
                           jnp.linspace(0, field_size, field_npix, endpoint=False)) # range of Y coordinates

coords = jnp.stack([xgrid, ygrid], axis=0)*u.deg
c = coords.reshape([2, -1]).T.to(u.rad)

# Cosmology 
cosmology = jc.Planck15(Omega_c=0.2589, sigma8=0.8159)

# %%
def convergence_Born(cosmo,
                     density_planes,
                     coords,
                     z_source):
  """
  Compute the Born convergence
  Args:
    cosmo: `Cosmology`, cosmology object.
    density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use 
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.
  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """
  # Compute constant prefactor:
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
  # Compute comoving distance of source galaxies
  r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

  convergence = 0
  ims = []

  for entry in density_planes:
    r = entry['r']; a = entry['a']; p = entry['plane']
    dx = entry['dx']; dz = entry['dz']
    # Normalize density planes
    density_normalization = dz * r / a
    p = (p - p.mean()) * constant_factor * density_normalization

    foo = coords * r / dx - 0.5
    coords_in = foo + ((plane_res - 1 - foo.max()) / 2)

    # Interpolate at the density plane coordinates
    im = map_coordinates(p, 
                         coords_in, 
                         order=1, mode="constant")

    convergence += im * jnp.clip(1. - (r / r_s), 0, 1000).reshape([-1, 1, 1])
    ims.append(im)
  

  return convergence, r_s, ims

def nbody(Omega_c=0.2589, sigma8=0.8159, seed=0):
    # Instantiates a cosmology with desired parameters

    # Planning out the scale factor stepping to extract desired lensplanes
    n_lens = int(box_size[-1] // lensplane_width)
    r = jnp.linspace(0., box_size[-1], n_lens+1)
    r_center = 0.5*(r[1:] + r[:-1])

    # Retrieve the scale factor corresponding to these distances
    a = jc.background.a_of_chi(cosmology, r)
    a_center = jc.background.a_of_chi(cosmology, r_center)

    # Then one step per lens plane
    stages = a_center[::-1]

    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmology, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    initial_conditions = linear_field(nc, box_size, pk_fn, seed=jax.random.PRNGKey(seed))

    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in nc]),axis=-1).reshape([-1,3])

    cosmo = jc.Planck15(Omega_c=Omega_c, sigma8=sigma8)

    # Initial displacement
    dx, p, f = lpt(cosmo, initial_conditions, particles, 0.1)

    # Evolve the simulation forward
    res = odeint(make_ode_fn(nc), [particles+dx, p], 
                 jnp.concatenate([jnp.atleast_1d(0.1), stages]), cosmo, rtol=1e-5, atol=1e-5)
    
    return res, a_center, r_center

def get_planes(res, a_center, r_center, plane_res=64):
    stages = a_center[::-1]    
    # Extract the lensplanes
    lensplanes = []
    for i in range(len(a_center)):
        dx = box_size[0]/plane_res
        dz = lensplane_width
        plane = density_plane(res[0][::-1][i],
                              nc,
                              (i+0.5)*lensplane_width/box_size[-1]*nc[-1],
                              width=lensplane_width/box_size[-1]*nc[-1],
                              plane_resolution=plane_res
                           )
        lensplanes.append({'r': r_center[i], 
                           'a': stages[::-1][i], 
                           'plane': plane,
                           'dx':dx,
                           'dz':dz})
    return lensplanes

# %%
res, a_center, r_center = nbody(seed=seed)

# %%
lensplanes = get_planes(res, a_center, r_center, plane_res=plane_res)

# %%
m, r_s, ims = convergence_Born(cosmology, 
                        lensplanes,
                        coords=jnp.array(c).T.reshape(2,field_npix,field_npix),
                        z_source=z_source)

# %%
def radial_profile(data):
  """
  Compute the radial profile of 2d image
  :param data: 2d image
  :return: radial profile
  """
  center = data.shape[0]/2
  y, x = jnp.indices((data.shape))
  r = jnp.sqrt((x - center)**2 + (y - center)**2)
  r = r.astype('int32')

  tbin = jnp.bincount(r.ravel(), data.ravel())
  nr = jnp.bincount(r.ravel())
  radialprofile = tbin / nr
  return radialprofile

def measure_power_spectrum(map_data, pixel_size):
  """
  measures power 2d data
  :param power: map (nxn)
  :param pixel_size: pixel_size (rad/pixel)
  :return: ell
  :return: power spectrum
  
  """
  data_ft = jnp.fft.fftshift(jnp.fft.fft2(map_data)) / map_data.shape[0]
  nyquist = int(map_data.shape[0]/2)
  power_spectrum_1d =  radial_profile(jnp.real(data_ft*jnp.conj(data_ft)))[:nyquist] * (pixel_size)**2
  k = jnp.arange(power_spectrum_1d.shape[0])
  ell = 2. * jnp.pi * k / pixel_size / 360
  return ell, power_spectrum_1d

# %%
resolution = 1 # arcmin/pixel
pixel_size = jnp.pi * resolution / 180. / 60. #rad/pixel

ps = measure_power_spectrum(m[-1], pixel_size)

jnp.save('../ps/ps_%03d.npy'%(args.seed), ps)