# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for TPU-specific interpret mode."""

import contextlib

from absl.testing import absltest
import numpy as np

import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec


# Run all tests with 8 CPU devices.
_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(8))

def tearDownModule():
  _exit_stack.close()

class InterpretTest(jtu.JaxTestCase):

  def test_right_permute_example(self):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jax.make_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    # Create an input array that shards the last dimension across
    # all devices.
    input_arr = jax.random.uniform(
      jax.random.key(0), (8, 128 * num_devices), dtype=jnp.float32)
    input_arr = jax.device_put(input_arr, sharding)

    def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem):
      my_id = lax.axis_index('x')
      left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
      right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))

      barrier_sem = pltpu.get_barrier_semaphore()
      def _body(ijk):
        i, (j, k) = ijk
        lax.cond(
          (i == 0) | (j == 0),
          lambda: pltpu.semaphore_signal(
            barrier_sem,
            device_id=(left_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH),
          lambda: pltpu.semaphore_signal(
            barrier_sem,
            device_id=(right_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH))
        return (i + 1, (j + 1, k + 1))
      lax.while_loop(lambda ijk: ijk[0] < 2, _body, (0, (0, 0)))
      pltpu.semaphore_wait(barrier_sem, 2)

      def _body2(i, a):
        remote_copy_op = pltpu.make_async_remote_copy(
          src_ref=input_ref,
          dst_ref=output_ref,
          send_sem=send_sem,
          recv_sem=recv_sem,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        remote_copy_op.start()
        remote_copy_op.wait()
        return i + 1, a + 1
      _ = lax.scan(_body2, 0, jnp.arange(4.0), unroll=2)

    out_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # TPUMemorySpace.ANY will (usually) place the tensor in HBM.
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        scratch_shapes=(
            # We allocate DMA semaphores in scratch memory.
            [pltpu.SemaphoreType.DMA] * 2
        ),
    )
    right_permute = pl.pallas_call(
        right_permute_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.TPUCompilerParams(collective_id=13),
        interpret=True,
        backend='mosaic_tpu',
    )
    # Wrap the kernel within a shard_map to call.
    pallas_result = jax.jit(
        shard_map.shard_map(
            right_permute,
            mesh=mesh,
            in_specs=partition,
            out_specs=partition,
            check_rep=False,
        )
    )(input_arr)

    # Compare Pallas result to XLA shard_map result.
    perm = tuple((src, (src + 1) % num_devices) for src in range(num_devices))
    xla_result = jax.jit(
        shard_map.shard_map(
            lambda x: lax.ppermute(x, 'x', perm),
            mesh=mesh, in_specs=partition, out_specs=partition)
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result)


  def test_all_gather_example(self):
    num_devices = jax.device_count()
    partition = P('x', None)
    mesh = jax.make_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    # Create an input array that shards the first dimension across
    # all devices.
    input_arr = jax.random.uniform(jax.random.key(0), (8 * num_devices, 128))
    input_arr = jax.device_put(input_arr, sharding)


    def all_gather_kernel(input_ref,
                          output_ref,
                          local_copy_sem,
                          send_sem,
                          recv_sems):
      outer_step = pl.program_id(0)
      my_id = lax.axis_index('x')
      right_neighbor = lax.rem(my_id + 1, num_devices)
      copy_slot = my_id - outer_step
      copy_slot = lax.rem(copy_slot + num_devices, num_devices)

      @pl.when(outer_step == 0)
      def _():
        local_copy_op = pltpu.make_async_copy(
          src_ref=input_ref,
          dst_ref=output_ref.at[my_id],
          sem=local_copy_sem,
        )
        local_copy_op.start()
        local_copy_op.wait()

      # Copy to our right neighbor.
      # Note that we will also be receiving data from our left neighbor,
      # but at `copy_slot-1` rather than `copy_slot`! This makes use of the fact
      # that the indices do not need to be symmetric between remote DMAs.
      remote_copy_op = pltpu.make_async_remote_copy(
        src_ref=output_ref.at[copy_slot],
        dst_ref=output_ref.at[copy_slot],
        send_sem=send_sem,
        recv_sem=recv_sems.at[outer_step],
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )
      remote_copy_op.start()
      remote_copy_op.wait()

    out_shape = jax.ShapeDtypeStruct((num_devices, 8, 128), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
        # TPUMemorySpace.ANY will (usually) place the tensor in HBM.
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      scratch_shapes=(
        # DMA semaphores are allocated in scratch memory.
        # We allocated one semaphore for a local HBM-VMEM copy,
        # and one for the remote send semaphore.
        [pltpu.SemaphoreType.DMA] * 2
        # We additionally allocate one receive semaphore per device.
        # This is to avoid situations where we have multiple
        # DMAs in flight, as we do not want to share a receive
        # semaphore between the DMAs.
        + [pltpu.SemaphoreType.DMA((num_devices-1,))]
      ),
      grid=(num_devices-1,)
    )

    all_gather = pl.pallas_call(
      all_gather_kernel,
      out_shape=out_shape,
      grid_spec=grid_spec,
      interpret=True,
      backend='mosaic_tpu',
    )

    # Wrap the kernel within a shard_map to call.
    pallas_result = jax.jit(
      shard_map.shard_map(
        all_gather,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_rep=False
      )
    )(input_arr)

    # Compare Pallas result to XLA shard_map result.
    xla_result = jax.jit(
      shard_map.shard_map(
        lambda x: lax.all_gather(x, 'x'),
        mesh=mesh, in_specs=partition, out_specs=partition
      )
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
