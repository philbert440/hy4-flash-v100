# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager


# CUDA allocator settings are process-global. Keep track of the active scope
# so nested startup controls restore it instead of restoring the environment.
_scoped_config: str | None = None


def effective_allocator_config() -> str:
    if _scoped_config is not None:
        return _scoped_config
    return os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "")
    )


@contextmanager
def scoped_allocator_config(
    setter: Callable[[str], None], config: str
) -> Iterator[None]:
    global _scoped_config
    previous_scope = _scoped_config
    previous_config = effective_allocator_config()
    setter(config)
    _scoped_config = config
    try:
        yield
    finally:
        _scoped_config = previous_scope
        setter(previous_config)
