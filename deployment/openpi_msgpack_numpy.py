"""MessagePack NumPy codec compatible with ``openpi_client.msgpack_numpy``.

X-WAM's original dependency, :mod:`msgpack_numpy`, uses a different wire
schema from OpenPI's vendored codec.  They cannot be mixed: an OpenPI ndarray
would otherwise arrive as an ordinary dictionary (and ``np.asarray`` then has
shape ``()``).  Keep this tiny protocol adapter in the server repository so
the robot-side OpenPI client is not a runtime dependency of the X-WAM venv.

``unpackb`` additionally accepts the PyPI msgpack-numpy schema.  That makes
manual Python clients using the old server implementation continue to work.
"""

import functools

import msgpack
import numpy as np


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    # OpenPI's codec (the schema used by WebsocketClientPolicy).
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    # Legacy PyPI msgpack-numpy schema, kept for backwards-compatible manual
    # clients.  Deployment payloads use ordinary numeric dtypes only.
    if obj.get(b"nd") is True:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"type"]), shape=obj[b"shape"])
    if obj.get(b"nd") is False:
        return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"type"]))[0]
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=_unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
