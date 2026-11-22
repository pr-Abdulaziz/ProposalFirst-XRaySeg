"""Rank-safe tqdm progress bars for MMEngine train, validation, and test loops."""

from typing import Any

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from tqdm.auto import tqdm

from mmdet.registry import HOOKS


@HOOKS.register_module()
class TqdmProgressHook(Hook):
    """Display one compact progress bar for the active MMEngine loop.

    Only rank zero renders a bar, so distributed jobs do not duplicate
    terminal output. The standard LoggerHook remains responsible for durable
    JSON/text metric logging; this hook is only a live progress display.
    """

    priority = "VERY_LOW"

    def __init__(
        self,
        enabled: bool = True,
        mininterval: float = 1.0,
        leave: bool = False,
    ) -> None:
        if mininterval <= 0:
            raise ValueError("mininterval must be positive.")
        self.enabled = enabled
        self.mininterval = mininterval
        self.leave = leave
        self._bar = None

    @property
    def _display(self) -> bool:
        return self.enabled and is_main_process()

    def _open(self, total: int, description: str) -> None:
        self._close()
        if not self._display:
            return
        self._bar = tqdm(
            total=total,
            desc=description,
            dynamic_ncols=True,
            leave=self.leave,
            mininterval=self.mininterval,
            unit="batch",
        )

    def _update(self, postfix: dict[str, Any] | None = None) -> None:
        if self._bar is None:
            return
        self._bar.update(1)
        if postfix:
            self._bar.set_postfix(postfix, refresh=False)

    def _close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def before_train_epoch(self, runner) -> None:
        max_epochs = getattr(runner, "max_epochs", None)
        epoch_label = f"{runner.epoch + 1}"
        if max_epochs is not None:
            epoch_label += f"/{max_epochs}"
        self._open(len(runner.train_dataloader), f"Train {epoch_label}")

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs: dict | None = None,
    ) -> None:
        loss = self._scalar(outputs.get("loss")) if outputs else None
        self._update({"loss": f"{loss:.4f}"} if loss is not None else None)

    def after_train_epoch(self, runner) -> None:
        self._close()

    def before_val_epoch(self, runner) -> None:
        self._open(len(runner.val_dataloader), "Validate")

    def after_val_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        self._update()

    def after_val_epoch(self, runner, metrics: dict | None = None) -> None:
        self._close()

    def before_test_epoch(self, runner) -> None:
        self._open(len(runner.test_dataloader), "Test")

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        self._update()

    def after_test_epoch(self, runner, metrics: dict | None = None) -> None:
        self._close()

    def after_run(self, runner) -> None:
        self._close()
