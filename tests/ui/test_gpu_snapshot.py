"""Cheap contract tests for the GPU snapshot boundary (no GL context)."""

from types import SimpleNamespace

import torch

from boost_and_broadside.ui.gpu_renderer import (
    MAX_FOG_CORES,
    MAX_OBSERVERS,
    GPUCamera,
    fog_uniform_payload,
    interpolate_toroidal_position,
    is_in_vision_range,
    map_instance_payload,
    packed_fog_uniform_payload,
    segment_clear_of_core,
    toroidal_delta,
)
from boost_and_broadside.ui.gpu_snapshot import (
    SnapshotCore,
    make_packed_render_snapshot,
    make_render_snapshot,
)


def _state(position: complex = 2 + 3j):
    return SimpleNamespace(
        num_envs=1,
        num_zones=1,
        step_count=torch.tensor([7]),
        ship_pos=torch.tensor([[position, 80 + 90j]]),
        ship_attitude=torch.tensor([[1 + 0j, 0 + 1j]]),
        ship_alive=torch.tensor([[True, False]]),
        ship_team_id=torch.tensor([[0, 1]]),
        ship_health=torch.tensor([[3.0, 0.0]]),
        bullet_active=torch.tensor([[[True, False], [False, True]]]),
        bullet_pos=torch.tensor([[[4 + 5j, 0j], [0j, 9 + 8j]]]),
        bullet_time=torch.tensor([[[3.0, 0.0], [0.0, 3.0]]]),
        map_center=torch.tensor([50 + 50j]),
        playable_boundary_radius=torch.tensor([40.0]),
        zone_pos=torch.tensor([[30 + 30j]]),
        zone_radius=torch.tensor([[10.0]]),
        zone_roles=torch.tensor([[2]]),
        zone_capture_progress=torch.tensor([[0.25]]),
        field_pos=torch.tensor([[70 + 70j]]),
        field_radius=torch.tensor([[8.0]]),
        field_transition_width=torch.tensor([[2.0]]),
    )


def test_snapshot_owns_immutable_scalar_content_not_live_tensors():
    state = _state()
    snapshot = make_render_snapshot(state, world_size=(100, 100))
    state.ship_pos[0, 0] = 77 + 77j
    assert snapshot.ships[0].current_position == (2.0, 3.0)
    assert snapshot.projectiles[3].current_position == (9.0, 8.0)
    assert snapshot.step == 7


def test_snapshot_carries_previous_transforms_and_active_projectiles_only():
    previous = make_render_snapshot(_state(1 + 1j), world_size=(100, 100))
    current = _state(3 + 4j)
    snapshot = make_render_snapshot(current, previous=previous, world_size=(100, 100))
    assert snapshot.ships[0].previous_position == (1.0, 1.0)
    assert sum(projectile.active for projectile in snapshot.projectiles) == 2
    assert snapshot.map_center == (50.0, 50.0)
    assert snapshot.playable_boundary_radius == 40.0
    assert snapshot.zones[0].capture_progress == 0.25
    assert snapshot.fog.opaque_cores[0].radius == 7.0


def test_gpu_camera_wraps_pan_across_toroidal_seam():
    camera = GPUCamera((100, 100), (100, 100), center=(2, 50))
    camera.pan_pixels(10, 0)
    assert camera.center == (92.0, 50.0)


def test_toroidal_projection_uses_the_nearest_image_across_a_seam():
    assert toroidal_delta((3, 50), (98, 50), (100, 100)) == (5.0, 0.0)


def test_toroidal_interpolation_crosses_the_seam_instead_of_the_world_center():
    assert interpolate_toroidal_position((98, 50), (2, 50), 0.5, (100, 100)) == (0.0, 50.0)


def test_team_visibility_masks_are_copied_and_enforced_by_perspective():
    visibility = SimpleNamespace(
        ship=torch.tensor([[[True, False], [False, True]]]),
        bullet=torch.tensor([[[[True, False], [False, False]], [[False, False], [False, True]]]]),
        vision_range=25.0,
    )
    snapshot = make_render_snapshot(
        _state(), world_size=(100, 100), visibility=visibility, zones_occlude=True
    )
    assert snapshot.ships[0].visibility_bits == 1
    assert snapshot.ships[1].visibility_bits == 2
    assert snapshot.visible_to("team0", snapshot.ships[0].visibility_bits)
    assert not snapshot.visible_to("team1", snapshot.ships[0].visibility_bits)
    assert snapshot.projectiles[3].visibility_bits == 2
    assert snapshot.fog.observer_positions[0] == ((2.0, 3.0),)
    assert snapshot.fog.vision_range == 25.0
    assert fog_uniform_payload(snapshot, "team0")["opaque_cores"][0].radius == 7.0
    assert len(fog_uniform_payload(snapshot, "team0")["opaque_cores"]) == 2
    assert fog_uniform_payload(snapshot, "full") is None
    team0_payload = fog_uniform_payload(snapshot, "team0")
    assert len(team0_payload["observer_data"]) == 2 * MAX_OBSERVERS
    assert len(team0_payload["core_data"]) == 3 * MAX_FOG_CORES
    assert team0_payload["observer_data"][:2] == (2.0, 3.0)
    assert team0_payload["observer_data"][2:] == (0.0,) * (2 * MAX_OBSERVERS - 2)
    team1_payload = fog_uniform_payload(snapshot, "team1")
    assert team1_payload["observer_positions"] == ()
    assert team1_payload["observer_data"] == (0.0,) * (2 * MAX_OBSERVERS)


def test_reused_projectile_ring_slot_snaps_its_previous_position():
    previous_state = _state()
    current = _state()
    previous_state.bullet_active[0, 0, 0] = False
    current.bullet_pos[0, 0, 0] = 80 + 80j
    previous = make_render_snapshot(previous_state, world_size=(100, 100))
    snapshot = make_render_snapshot(current, previous=previous, world_size=(100, 100))
    assert snapshot.projectiles[0].previous_position == (80.0, 80.0)


def test_snapshot_performs_one_host_transfer(monkeypatch):
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    make_render_snapshot(_state(), world_size=(100, 100))
    assert calls == 1


def test_packed_snapshot_matches_reference_entity_values_and_owns_transfer():
    previous_state, current = _state(1 + 1j), _state(3 + 4j)
    previous = make_packed_render_snapshot(previous_state, world_size=(100, 100))
    packed = make_packed_render_snapshot(current, world_size=(100, 100))
    reference = make_render_snapshot(
        current,
        previous=make_render_snapshot(previous_state, world_size=(100, 100)),
        world_size=(100, 100),
    )
    current.ship_pos[0, 0] = 90 + 90j
    assert packed.ship_instances(previous)[0, :4].tolist() == [1.0, 1.0, 3.0, 4.0]
    assert packed.step == 7
    assert packed.projectile_instances(previous)[1, :4].tolist() == list(
        reference.projectiles[3].previous_position + reference.projectiles[3].current_position
    )


def test_packed_snapshot_performs_one_host_transfer(monkeypatch):
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    make_packed_render_snapshot(_state(), world_size=(100, 100))
    assert calls == 1


def test_packed_projectile_instances_filter_inactive_slots_and_snap_reuse():
    previous_state, current_state = _state(), _state()
    previous_state.bullet_active[0, 0, 0] = False
    current_state.bullet_pos[0, 0, 0] = 80 + 80j
    previous = make_packed_render_snapshot(previous_state, world_size=(100, 100))
    current = make_packed_render_snapshot(current_state, world_size=(100, 100))
    rows = current.projectile_instances(previous)
    assert rows.shape == (2, 6)
    assert rows[0, :4].tolist() == [80.0, 80.0, 80.0, 80.0]
    assert rows.is_contiguous() and rows.dtype == torch.float32
    assert rows.numpy().nbytes == rows.numel() * 4


def test_packed_map_and_fog_payloads_are_renderer_buffer_compatible():
    visibility = SimpleNamespace(
        ship=torch.tensor([[[True, False], [False, True]]]),
        bullet=torch.tensor([[[[True, False], [False, False]], [[False, False], [False, True]]]]),
        vision_range=25.0,
    )
    packed = make_packed_render_snapshot(
        _state(), world_size=(100, 100), visibility=visibility, zones_occlude=True
    )
    rows = packed.map_instances()
    fog = packed_fog_uniform_payload(packed, "team0")
    assert rows.tolist() == [
        [70.0, 70.0, 7.0, 0.0, 0.0],
        [30.0, 30.0, 10.0, 1.0, 2.0],
        [50.0, 50.0, 40.0, 2.0, 0.0],
    ]
    assert rows.is_contiguous() and rows.dtype == torch.float32
    assert fog is not None
    assert (fog.observer_count, fog.core_count) == (1, 2)
    assert fog.observer_data.shape == (2 * MAX_OBSERVERS,)
    assert fog.core_data.shape == (3 * MAX_FOG_CORES,)
    assert fog.observer_data.numpy().nbytes == 2 * MAX_OBSERVERS * 4
    assert fog.observer_data[:2].tolist() == [2.0, 3.0]
    assert packed_fog_uniform_payload(packed, "full") is None


def test_exact_toroidal_segment_core_semantics_preserve_touch_and_shared_inside():
    core = SnapshotCore((0.0, 50.0), 10.0)
    assert segment_clear_of_core((90.0, 50.0), (10.0, 50.0), core, (100.0, 100.0)) is False
    assert segment_clear_of_core((90.0, 60.0), (10.0, 60.0), core, (100.0, 100.0))
    assert segment_clear_of_core((98.0, 50.0), (2.0, 50.0), core, (100.0, 100.0))
    assert is_in_vision_range((98.0, 50.0), (3.0, 50.0), 5.0, (100.0, 100.0))


def test_map_instance_payload_batches_fields_zones_and_boundary():
    snapshot = make_render_snapshot(_state(), world_size=(100, 100), zones_occlude=True)
    rows = map_instance_payload(snapshot)
    assert [row[3] for row in rows] == [0.0, 1.0, 2.0]
    assert rows[1][4] == 2.0
    assert rows[2][:3] == (50.0, 50.0, 40.0)
