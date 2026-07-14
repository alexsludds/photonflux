"""Deck-centric circuit builder.

Raw SPICE stays first-class (the decks in this repo are written to be read),
but compiled Verilog-A devices, PDK libraries, and options get real APIs so
the boilerplate and the OSDI bookkeeping disappear.

A `Circuit` is a list of cards plus the set of OSDI modules its devices
need; the Engine loads those modules before the netlist automatically.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from .compiler import VaModule, check_module_name

__all__ = ["Circuit"]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        # repr = shortest round-trip form: high-Q photonics needs parameters
        # (lambda_nm, ...) exact to sub-pm, where "%g"'s 6 digits truncate.
        # float() first: np.float64 is a float subclass but reprs as
        # "np.float64(...)" on numpy >= 2.
        return repr(float(v))
    return str(v)


class Circuit:
    def __init__(self, title: str = "photonflux circuit"):
        self.title = title
        self._cards: list[str] = []
        self._model_cards: list[str] = []
        self._libs: list[str] = []
        self._includes: list[str] = []
        self._options: dict[str, Any] = {}
        self._osdi: dict[str, Path] = {}   # module name -> osdi path
        self._names: set[str] = set()

    # ------------------------------------------------------------ raw cards
    def raw(self, *blocks: str) -> "Circuit":
        """Append raw SPICE cards (multiline strings are dedented)."""
        for b in blocks:
            self._cards.extend(textwrap.dedent(b).strip("\n").splitlines())
        return self

    def __iadd__(self, block: str) -> "Circuit":
        return self.raw(block)

    def lib(self, path: str | Path, section: str = "tt") -> "Circuit":
        self._libs.append(f'.lib "{path}" {section}')
        return self

    def include(self, path: str | Path) -> "Circuit":
        self._includes.append(f'.include "{path}"')
        return self

    def options(self, **kw: Any) -> "Circuit":
        self._options.update(kw)
        return self

    # ------------------------------------------------------ Verilog-A devices
    def device(
        self,
        module: VaModule,
        name: str,
        *nodes: str,
        **params: Any,
    ) -> str:
        """Instantiate a compiled Verilog-A device.

        Emits a dedicated `.model` card carrying `params` plus the OSDI
        instance card (ngspice gives OSDI devices the `N` prefix). Returns
        the instance name.
        """
        check_module_name(module.name, module.va)
        if name in self._names:
            raise ValueError(f"duplicate device name {name!r}")
        self._names.add(name)

        unknown = set(map(str.lower, params)) - set(map(str.lower, module.params))
        if unknown:
            raise ValueError(
                f"{module.name} has no parameter(s) {sorted(unknown)}; "
                f"available: {sorted(module.params)}"
            )

        self._osdi[module.name] = module.osdi
        model_id = f"mod_{name}"
        ptxt = " ".join(f"{k.lower()}={_fmt(v)}" for k, v in params.items())
        self._model_cards.append(
            f".model {model_id} {module.name} ({ptxt})" if ptxt
            else f".model {model_id} {module.name}"
        )
        self._cards.append(f"N{name} {' '.join(nodes)} {model_id}")
        return f"N{name}"

    # ------------------------------------------------------------ assembly
    @property
    def osdi_files(self) -> list[Path]:
        return list(self._osdi.values())

    def build(self) -> str:
        parts: list[str] = [f"* {self.title}"]
        if self._options:
            parts.append(
                ".options " + " ".join(f"{k}={_fmt(v)}" for k, v in self._options.items())
            )
        parts += self._libs + self._includes + self._model_cards + self._cards
        parts.append(".end")
        return "\n".join(parts) + "\n"

    def __str__(self) -> str:
        return self.build()
