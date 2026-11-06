"""Record the PIDray data scope in every saved checkpoint."""

from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class PIDrayCheckpointProvenanceHook(Hook):
    """Attach split provenance so incompatible initializations can be rejected."""

    priority = "VERY_LOW"

    def __init__(self, training_scope: str, annotation_file: str) -> None:
        if training_scope != "official_full_train":
            raise ValueError(
                "training_scope must be 'official_full_train'."
            )
        self.training_scope = training_scope
        self.annotation_file = annotation_file

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        meta = checkpoint.setdefault("meta", {})
        meta["pidray_training_scope"] = self.training_scope
        meta["pidray_annotation_file"] = self.annotation_file
