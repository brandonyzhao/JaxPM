import os

os.environ["EQX_ON_ERROR"] = "nan"  # avoid an allgather caused by diffrax
import jax

jax.distributed.initialize()

rank = jax.process_index()
size = jax.process_count()

import argparse
import time

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from cupy.cuda.nvtx import RangePop, RangePush
from diffrax import (ConstantStepSize, Dopri5, LeapfrogMidpoint, ODETerm,
                     PIDController, SaveAt, Tsit5, diffeqsolve)
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpm.kernels import interpolate_power_spectrum
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import linear_field, lpt, make_ode_fn


def chrono_fun(fun, *args):
    start = time.perf_counter()
    out = fun(*args)
    out[0].block_until_ready()
    end = time.perf_counter()
    return out, end - start


def run_simulation(mesh_shape,
                   box_size,
                   halo_size,
                   solver_choice,
                   iterations,
                   pdims=None):

    @jax.jit
    def simulate(omega_c, sigma8):
        # Create a small function to generate the matter power spectrum
        k = jnp.logspace(-4, 1, 128)
        pk = jc.power.linear_matter_power(
            jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
        pk_fn = lambda x: interpolate_power_spectrum(x, k, pk)

        # Create initial conditions
        initial_conditions = linear_field(mesh_shape,
                                          box_size,
                                          pk_fn,
                                          seed=jax.random.PRNGKey(0))

        # Create particles
        cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
        dx, p, _ = lpt(cosmo, initial_conditions, 0.1, halo_size=halo_size)

        if solver_choice == "Dopri5":
            solver = Dopri5()
        elif solver_choice == "LeapfrogMidpoint":
            solver = LeapfrogMidpoint()
        elif solver_choice == "Tsit5":
            solver = Tsit5()
        elif solver_choice == "lpt":
            lpt_field = cic_paint_dx(dx, halo_size=halo_size)
            return lpt_field, {"num_steps": 0, "Solver": "LPT"}
        else:
            raise ValueError(
                "Invalid solver choice. Use 'Dopri5' or 'LeapfrogMidpoint'.")
        # Evolve the simulation forward
        ode_fn = make_ode_fn(mesh_shape, halo_size=halo_size)
        term = ODETerm(
            lambda t, state, args: jnp.stack(ode_fn(state, t, args), axis=0))

        stepsize_controller = PIDController(rtol=1e-4, atol=1e-4)
        res = diffeqsolve(term,
                          solver,
                          t0=0.1,
                          t1=1.,
                          dt0=0.01,
                          y0=jnp.stack([dx, p], axis=0),
                          args=cosmo,
                          saveat=SaveAt(t1=True),
                          stepsize_controller=stepsize_controller)

        # Return the simulation volume at requested
        state = res.ys[-1]
        final_field = cic_paint_dx(state[0], halo_size=halo_size)

        return final_field, res.stats

    def run():
        # Warm start
        times = []
        RangePush("warmup")
        (final_field, stats), warmup_time = chrono_fun(simulate, 0.32, 0.8)
        RangePop()
        sync_global_devices("warmup")
        for i in range(iterations):
            RangePush(f"sim iter {i}")
            (final_field, stats), sim_time = chrono_fun(simulate, 0.32, 0.8)
            RangePop()
            times.append(sim_time)
        return stats, warmup_time, times, final_field

    if jax.device_count() > 1:
        devices = mesh_utils.create_device_mesh(pdims)
        mesh = Mesh(devices.T, axis_names=('x', 'y'))
        with mesh:
            # Warm start
            stats, warmup_time, times, final_field = run()
    else:
        stats, warmup_time, times, final_field = run()

    return stats, warmup_time, times, final_field


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='JAX Cosmo Simulation Benchmark')
    parser.add_argument('-m',
                        '--mesh_size',
                        type=int,
                        help='Mesh size',
                        required=True)
    parser.add_argument('-b',
                        '--box_size',
                        type=float,
                        help='Box size',
                        required=True)
    parser.add_argument('-p',
                        '--pdims',
                        type=str,
                        help='Processor dimensions',
                        default=None)
    parser.add_argument('-hs',
                        '--halo_size',
                        type=int,
                        help='Halo size',
                        required=True)
    parser.add_argument('-s',
                        '--solver',
                        type=str,
                        help='Solver',
                        choices=[
                            "Dopri5", "dopri5", "d5", "Tsit5", "tsit5", "t5",
                            "LeapfrogMidpoint", "leapfrogmidpoint", "lfm",
                            "lpt"
                        ],
                        required=True)
    parser.add_argument('-i',
                        '--iterations',
                        type=int,
                        help='Number of iterations',
                        default=10)
    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        help='Output path',
                        default=".")
    parser.add_argument('-f',
                        '--save_fields',
                        action='store_true',
                        help='Save fields')

    args = parser.parse_args()

    mesh_size = args.mesh_size
    box_size = [args.box_size] * 3
    halo_size = args.halo_size
    solver_choice = args.solver
    iterations = args.iterations
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    match solver_choice:
        case "Dopri5", "dopri5", "d5":
            solver_choice = "Dopri5"
        case "Tsit5", "tsit5", "t5":
            solver_choice = "Tsit5"
        case "LeapfrogMidpoint", "leapfrogmidpoint", "lfm":
            solver_choice = "LeapfrogMidpoint"
        case "lpt":
            solver_choice = "lpt"
        case _:
            raise ValueError(
                "Invalid solver choice. Use 'Dopri5', 'Tsit5', 'LeapfrogMidpoint' or 'lpt"
            )

    if args.pdims:
        pdims = tuple(map(int, args.pdims.split("x")))
    else:
        pdims = (1, 1)

    mesh_shape = [mesh_size] * 3

    stats, warmup_time, times, final_field = run_simulation(mesh_shape,
                                                            box_size,
                                                            halo_size,
                                                            solver_choice,
                                                            iterations,
                                                            pdims=pdims)

    # Write benchmark results to CSV
    # RANK SIZE MESHSIZE BOX HALO SOLVER NUM_STEPS JITTIME MIN MAX MEAN STD
    times = np.array(times)
    jit_in_ms = (warmup_time * 1000)
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    mean_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000

    with open(f"{output_path}/jax_pm_benchmark.csv", 'a') as f:
        f.write(
            f"{rank},{size},{mesh_size},{box_size[0]},{halo_size},{solver_choice},{stats['num_steps']},{jit_in_ms},{min_time},{max_time},{mean_time},{std_time}\n"
        )

    # Save the final field
    nb_gpus = jax.device_count()
    pdm_str = f"{pdims[0]}x{pdims[1]}"
    field_folder = f"{output_path}/final_field/{nb_gpus}/{mesh_size}_{int(box_size[0])}/{pdm_str}/{solver_choice}/{halo_size}"
    os.makedirs(field_folder, exist_ok=True)
    with open(f'{field_folder}/jaxpm.log', 'w') as f:
        f.write(f"Args: {args}\n")
        f.write(f"JIT time: {jit_in_ms:.4f} ms\n")
        f.write(f"Min time: {min_time:.4f} ms\n")
        f.write(f"Max time: {max_time:.4f} ms\n")
        f.write(f"Mean time: {mean_time:.4f} ms\n")
        f.write(f"Std time: {std_time:.4f} ms\n")
        f.write(f"Stats: {stats}\n")
    if args.save_fields:
        np.save(f'{field_folder}/final_field_0_{rank}.npy',
                final_field.addressable_data(0))

    print(f"Finished! Warmup time: {warmup_time:.4f} seconds")
    print(f"mean times: {np.mean(times):.4f}")
    print(f"Stats {stats}")
    print(f"Saving to {output_path}/jax_pm_benchmark.csv")
    print(f"Saving field and logs in {field_folder}")