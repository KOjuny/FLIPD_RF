import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Tuple

from omegaconf import Container, DictConfig, ListConfig, OmegaConf
from omegaconf._utils import Marker
from omegaconf.basecontainer import BaseContainer
from omegaconf.errors import ConfigKeyError

_DEFAULT_SELECT_MARKER_: Any = Marker("_DEFAULT_SELECT_MARKER_")
MAX_RESOLVE_DEPTH = 7


class Tee:
    """This class allows for redirecting of stdout and stderr"""

    def __init__(self, primary_file, secondary_file):
        self.primary_file = primary_file
        self.secondary_file = secondary_file

        self.encoding = self.primary_file.encoding

    # TODO: Should redirect all attrs to primary_file if not found here.
    def isatty(self):
        return self.primary_file.isatty()

    def fileno(self):
        return self.primary_file.fileno()

    def write(self, data):
        # We get problems with ipdb if we don't do this:
        if isinstance(data, bytes):
            data = data.decode()

        self.primary_file.write(data)
        self.secondary_file.write(data)

    def flush(self):
        self.primary_file.flush()
        self.secondary_file.flush()


def link_output_streams(artifact_dir: Path):
    out_file = open(artifact_dir / "stdout.txt", "a")
    err_file = open(artifact_dir / "stderr.txt", "a")
    sys.stdout = Tee(
        primary_file=sys.stdout,
        secondary_file=out_file,
    )
    sys.stderr = Tee(
        primary_file=sys.stderr,
        secondary_file=err_file,
    )


def setup_root():
    # Add the root directory to path (no matter where the script is run from)
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    sys.path.append(".")


def _get_and_validate_dict_input(
    key: str,
    parent: BaseContainer,
    resolver_name: str,
) -> DictConfig:
    from omegaconf._impl import select_value

    if not isinstance(key, str):
        raise TypeError(
            f"`{resolver_name}` requires a string as input, but obtained `{key}` "
            f"of type: {type(key).__name__}"
        )

    in_dict = select_value(
        parent,
        key,
        throw_on_missing=True,
        absolute_key=True,
        default=_DEFAULT_SELECT_MARKER_,
    )

    if in_dict is _DEFAULT_SELECT_MARKER_:
        raise ConfigKeyError(f"Key not found: '{key}'")

    if not isinstance(in_dict, DictConfig):
        raise TypeError(
            f"`{resolver_name}` cannot be applied to objects of type: " f"{type(in_dict).__name__}"
        )

    return in_dict


def _recursive_assert(condition: bool, msg: str, current_path: List):
    assert condition, f"{current_path}: {msg}"


def _get_transform_idx(key: str, current_path: List) -> int:
    # check if key starts with "t"
    _recursive_assert(
        key.startswith("t"),
        f"Key {key} must start with 't' for data transforms",
        current_path,
    )
    # check if the rest of the key is a number
    _recursive_assert(
        key[1:].isdigit(),
        f"Key {key} must end with a number for data transforms",
        current_path,
    )
    return int(key[1:])


def _resolve_dict(in_dict: DictConfig, current_path: List | None = None) -> List[Tuple[int, Any]]:
    """
    This script adds list manipulation capabilities to the omegaconf resolver!

    Args:
        in_dict: The input dictionary to be resolved
        current_path: The sequence of keys that led to the current dictionary
    Returns:
        A list of tuples where the first element is the index of the transform and the second element is the transform itself
    """
    current_path = current_path or []
    _recursive_assert(
        len(current_path) < MAX_RESOLVE_DEPTH,
        f"Max depth of {MAX_RESOLVE_DEPTH} exceeded!",
        current_path,
    )
    prepend_transforms = []
    all_transforms = []
    append_transforms = []
    seen_prepend = False
    seen_append = False
    seen_clear = False
    delete_idx = set()
    insert_dict = {}
    for key, value in in_dict.items():
        if key == "prepend":
            _recursive_assert(not seen_prepend, "prepend can only be specified once", current_path)
            _recursive_assert(
                isinstance(value, DictConfig),
                "prepend must be a dictionary",
                current_path,
            )
            seen_prepend = True
            prepend_transforms = _resolve_dict(value, current_path=current_path + [key])
        elif key == "append":
            _recursive_assert(not seen_append, "append can only be specified once", current_path)
            _recursive_assert(
                isinstance(value, DictConfig),
                "append must be a dictionary",
                current_path,
            )
            seen_append = True
            append_transforms = _resolve_dict(value, current_path=current_path + [key])
        elif key == "clear":
            _recursive_assert(not seen_clear, "clear can only be specified once", current_path)
            _recursive_assert(isinstance(value, bool), "clear must be a boolean", current_path)
            seen_clear = True
        elif key.startswith("insert_"):
            _recursive_assert(
                isinstance(value, DictConfig),
                "insert must be a dictionary",
                current_path,
            )
            # chceck value to only have one key
            real_key = key[len("insert_") :]
            insert_dict[_get_transform_idx(real_key, current_path)] = _resolve_dict(
                value, current_path=current_path + [key]
            )
        elif key == "delete":
            _recursive_assert(
                isinstance(value, str), "delete must be a string t{idx}", current_path
            )
            delete_idx.add(_get_transform_idx(value, current_path))
        else:
            all_transforms.append((_get_transform_idx(key, current_path), value))
    # sort all the transforms by key
    all_transforms = sorted(all_transforms, key=lambda x: x[0])

    # check if all the first values in all_transforms is unique
    seen = set()
    for idx, _ in all_transforms:
        _recursive_assert(
            idx not in seen,
            f"Transform index {idx} is repeated. Make sure that all the transform indices are unique",
            current_path,
        )
        seen.add(idx)

    # handle deletion
    filtered_values = []
    for idx, value in enumerate(all_transforms):
        if idx in delete_idx:
            delete_idx.remove(idx)
        else:
            filtered_values.append(value)
    _recursive_assert(
        len(delete_idx) == 0,
        f"delete failed! The following transforms are not available to remove! {delete_idx}",
        current_path,
    )
    all_transforms = filtered_values

    # handle insertion
    for idx in insert_dict.keys():
        # check if idx is valid and exists in all_transforms
        valid = False
        for transform_idx, _ in all_transforms:
            if transform_idx == idx:
                valid = True
                break
        _recursive_assert(
            valid,
            f"Insert index {idx} is invalid because it is either not available or it is deleted!",
            current_path,
        )
    expanded_transforms = []
    for idx, value in all_transforms:
        expanded_transforms.append((idx, value))
        if idx in insert_dict:
            expanded_transforms += insert_dict[idx]
    all_transforms = expanded_transforms

    # handle prepending
    ret = prepend_transforms
    if not seen_clear:
        ret += all_transforms
    ret += append_transforms
    return list(ret)


def resolve_recursive_list(key: str, _root_: BaseContainer, _parent_: Container) -> ListConfig:
    assert isinstance(_parent_, BaseContainer)
    in_dict = _get_and_validate_dict_input(
        key, parent=_parent_, resolver_name="resolve_data_transform"
    )
    all_transforms_sorted = _resolve_dict(in_dict=in_dict)
    ret = ListConfig([t[1] for t in all_transforms_sorted])
    element_type: Any = in_dict._metadata.element_type
    ret._metadata.element_type = element_type
    ret._metadata.ref_type = List[element_type]
    ret._set_parent(_parent_)
    return ret


@contextmanager
def custom_resolvers():
    OmegaConf.register_new_resolver("resolve_recursive_list", resolve_recursive_list)
    try:
        yield
    finally:
        OmegaConf.clear_resolver("resolve_recursive_list")
