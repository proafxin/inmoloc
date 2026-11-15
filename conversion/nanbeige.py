from __future__ import annotations

from conversion.base import ModelBase, gguf, logger
from conversion.llama import LlamaModel


@ModelBase.register("NanbeigeForCausalLM")
@ModelBase.example("Nanbeige/Nanbeige4.2-3B")
class NanbeigeModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.NANBEIGE
    undo_permute = True

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.hparams

        n_loops = int(hparams.get("num_loops", 1) or 1)
        n_loops = max(n_loops, 1)
        self.gguf_writer.add_num_loops(n_loops)
        logger.info(f"gguf: num_loops = {n_loops}")

        skip_loop_final_norm = bool(hparams.get("skip_loop_final_norm", False))
        self.gguf_writer.add_skip_loop_final_norm(skip_loop_final_norm)
        logger.info(f"gguf: skip_loop_final_norm = {skip_loop_final_norm}")
