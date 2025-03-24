from typing import TypedDict


ModuleMeta = TypedDict(
    "ModuleMeta",
    {
        "name": str,
        "drop_set": set[int],
    },
)
