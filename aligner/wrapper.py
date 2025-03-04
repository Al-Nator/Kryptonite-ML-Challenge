from transformers import PreTrainedModel
from transformers import PretrainedConfig
from omegaconf import OmegaConf
from .aligners import get_aligner
import yaml

class ModelConfig(PretrainedConfig):

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.conf = dict(yaml.safe_load(open('aligner/pretrained_model/model.yaml')))


class CVLFaceAlignmentModel(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        model_conf = OmegaConf.create(cfg.conf)
        self.model = get_aligner(model_conf)
        self.model.load_state_dict_from_path('aligner/pretrained_model/model.pt')

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
