from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch
from vllm.distributed.parallel_state import TensorMetadata
from vllm.v1.core.sched.output import HiddenChannelType

import vllm_ascend.distributed.parallel_state as parallel_state
from vllm_ascend.distributed.parallel_state import (
    EdgeCloudTensorMeta,
    edge_cloud_broadcast_recv_scheduled_draft,
    edge_cloud_irecv_tensor_dict_on_hidden_channel,
    edge_cloud_send_tensor_dict,
    edge_cloud_send_tensor_dict_scheduled_draft,
)


@pytest.fixture(autouse=True)
def edge_cloud_tensor_meta():
    parallel_state._EDGE_CLOUD_TENSOR_META = EdgeCloudTensorMeta(
        metadata_list=[
            ("hidden_states", TensorMetadata("cpu", torch.float32, (0, 4))),
            ("residual", TensorMetadata("cpu", torch.float32, (0, 4))),
        ],
        tensor_keys=["hidden_states", "residual"],
        hc_mult=1,
    )
    yield
    parallel_state._EDGE_CLOUD_TENSOR_META = None


def _pp_group_with_hidden_channels():
    device_group = object()
    alt_device_group = object()
    prefill2_device_group = object()
    draft_device_group = object()
    group_by_channel = {
        HiddenChannelType.PREFILL_1: device_group,
        HiddenChannelType.DECODE: alt_device_group,
        HiddenChannelType.PREFILL_2: prefill2_device_group,
        HiddenChannelType.DRAFT: draft_device_group,
    }

    pp_group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        ranks=[0, 1],
        device_group=device_group,
        alt_device_group=alt_device_group,
        isend_tensor_dict_on_hidden_channel=Mock(
            side_effect=AssertionError("metadata path must not be used")
        ),
        irecv_tensor_dict_on_hidden_channel=Mock(
            side_effect=AssertionError("metadata path must not be used")
        ),
    )

    def hidden_channel_groups(channel):
        return group_by_channel[channel], object()

    pp_group._hidden_channel_groups = hidden_channel_groups
    return pp_group, group_by_channel


@pytest.mark.parametrize(
    "channel",
    [
        HiddenChannelType.PREFILL_1,
        HiddenChannelType.DECODE,
        HiddenChannelType.PREFILL_2,
        HiddenChannelType.DRAFT,
    ],
)
def test_edge_cloud_send_uses_hidden_channel_without_metadata(channel):
    pp_group, group_by_channel = _pp_group_with_hidden_channels()
    tensor_dict = {
        "hidden_states": torch.ones(6, 4),
        "residual": torch.ones(6, 4),
    }

    with patch.object(parallel_state, "get_pp_group", return_value=pp_group), \
            patch("torch.distributed.isend", return_value=Mock()) as mock_isend:
        handles = edge_cloud_send_tensor_dict(
            tensor_dict,
            channel=channel,
            num_tokens=3,
        )

    assert len(handles) == 2
    assert mock_isend.call_count == 2
    pp_group.isend_tensor_dict_on_hidden_channel.assert_not_called()
    for call in mock_isend.call_args_list:
        sent_tensor = call.args[0]
        assert sent_tensor.shape == (3, 4)
        assert call.kwargs["group"] is group_by_channel[channel]


def test_edge_cloud_irecv_uses_precomputed_meta_without_metadata():
    pp_group, group_by_channel = _pp_group_with_hidden_channels()
    pp_group.rank_in_group = 1

    with patch.object(parallel_state, "get_pp_group", return_value=pp_group), \
            patch("torch.distributed.is_initialized", return_value=True), \
            patch("torch.distributed.irecv", return_value=Mock()) as mock_irecv:
        tensor_dict, handles, postprocess = edge_cloud_irecv_tensor_dict_on_hidden_channel(
            channel=HiddenChannelType.PREFILL_2,
            num_tokens=5,
        )

    assert len(handles) == 2
    assert postprocess == []
    assert tensor_dict["hidden_states"].shape == (5, 4)
    assert tensor_dict["residual"].shape == (5, 4)
    pp_group.irecv_tensor_dict_on_hidden_channel.assert_not_called()
    for call in mock_irecv.call_args_list:
        recv_tensor = call.args[0]
        assert recv_tensor.shape == (5, 4)
        assert call.kwargs["group"] is group_by_channel[HiddenChannelType.PREFILL_2]


def test_edge_cloud_hidden_channel_fallback_without_extra_groups():
    pp_group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        ranks=[0, 1],
        device_group=object(),
        alt_device_group=object(),
    )
    tensor_dict = {
        "hidden_states": torch.ones(2, 4),
        "residual": torch.ones(2, 4),
    }

    with patch.object(parallel_state, "get_pp_group", return_value=pp_group), \
            patch("torch.distributed.isend", return_value=Mock()) as mock_isend:
        edge_cloud_send_tensor_dict(
            tensor_dict,
            channel=HiddenChannelType.PREFILL_1,
            num_tokens=2,
        )
        assert mock_isend.call_args_list[-1].kwargs["group"] is pp_group.device_group

        edge_cloud_send_tensor_dict(
            tensor_dict,
            channel=HiddenChannelType.DECODE,
            num_tokens=2,
        )
        assert mock_isend.call_args_list[-1].kwargs["group"] is pp_group.alt_device_group

        with pytest.raises(RuntimeError, match="PREFILL_2 hidden channel"):
            edge_cloud_send_tensor_dict(
                tensor_dict,
                channel=HiddenChannelType.PREFILL_2,
                num_tokens=2,
            )

        with pytest.raises(RuntimeError, match="draft hidden channel"):
            edge_cloud_send_tensor_dict(
                tensor_dict,
                channel=HiddenChannelType.DRAFT,
                num_tokens=2,
            )


def test_scheduled_draft_defaults_to_dedicated_draft_channel():
    send = Mock(return_value=[])
    recv = Mock(return_value=({}, [], []))
    pp_group = SimpleNamespace(
        world_size=2,
        isend_tensor_dict_on_hidden_channel=send,
        irecv_tensor_dict_on_hidden_channel=recv,
    )
    tp_group = SimpleNamespace(
        broadcast_object=Mock(return_value=None),
    )
    stream_context = Mock(
        side_effect=lambda *args, **kwargs: nullcontext()
    )

    with patch.object(parallel_state, "get_pp_group", return_value=pp_group), \
            patch.object(parallel_state, "get_tp_group", return_value=tp_group), \
            patch.object(
                parallel_state,
                "_hidden_channel_stream_ctx",
                new=stream_context,
            ), \
            patch("torch.distributed.is_initialized", return_value=True):
        assert edge_cloud_send_tensor_dict_scheduled_draft({}) == []
        tensor_dict, handles, postprocess = (
            edge_cloud_broadcast_recv_scheduled_draft()
        )

    assert tensor_dict == {}
    assert handles == []
    assert len(postprocess) == 1

    send.assert_called_once_with(
        {},
        dst=None,
        channel=HiddenChannelType.DRAFT,
    )
    recv.assert_called_once_with(
        src=None,
        channel=HiddenChannelType.DRAFT,
    )
    assert stream_context.call_args_list == [
        call(HiddenChannelType.DRAFT, wait_for_default=True),
        call(HiddenChannelType.DRAFT, wait_for_default=False),
    ]


def test_scheduled_draft_fails_without_hidden_channel_patch():
    pp_group = SimpleNamespace(world_size=2)

    with patch.object(parallel_state, "get_pp_group", return_value=pp_group):
        with pytest.raises(RuntimeError, match="dedicated hidden-channel"):
            edge_cloud_send_tensor_dict_scheduled_draft({})
