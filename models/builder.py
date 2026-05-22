import os
from functools import partial
import timm
import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple
from .timm_wrapper import TimmCNNEncoder
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms


class ConvStem(nn.Module):
    """CTransPath용 커스텀 Patch Embed 레이어 (CNN 기반).
    Ref: https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    elif model_name == 'ctranspath':
        # trident 의 ctranspath() 를 그대로 사용 (timm_ctp 기반).
        # 이전 버전 (timm + 자체 ConvStem) 은 forward 가 잘못 정의되어 모든 패치가
        # 거의 같은 feature 를 출력 (cos sim 0.99+) — 사실상 random representation.
        # trident 의 timm_ctp 기반 구현은 정상 작동 (cos sim 0.7-0.9).
        import sys as _sys
        from huggingface_hub import hf_hub_download
        _trident_path = '/data/CLAM-master/trident/trident/patch_encoder_models/model_zoo/ctranspath'
        if _trident_path not in _sys.path:
            _sys.path.insert(0, _trident_path)
        from ctran import ctranspath  # type: ignore
        import torch.nn as nn

        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        ckpt_path = hf_hub_download(
            repo_id='MahmoodLab/hest-bench',
            repo_type='dataset',
            filename='CHIEF_CTransPath.pth',
            subfolder='fm_v1/ctranspath',
            cache_dir='/data/CLAM-master/checkpoints/ctranspath',
        )
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        # attn_mask 제거 (forward 에서 동적 생성)
        state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"CTransPath loaded (trident/timm_ctp): "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if len(unexpected) > 5:
            raise RuntimeError(
                f"CTransPath weights load failed: {len(unexpected)} unexpected keys"
            )
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms