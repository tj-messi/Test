import os, sys, re, yaml, contextlib
from pathlib import Path     
from functools import partial
from typing import Optional
from ..core import register
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized
from ..backbone.common import FrozenBatchNorm2d

import torch 
import torch.nn as nn 

from engine.logger_module import get_logger
from engine.misc.ov_text_cache import load_text_cache_payload, summarize_text_cache
from engine.ovdeim.text_adapter import TextAdapter   

from engine.backbone.hgnetv2 import HGNetv2, StemBlock, HG_Stage, filter_loadable_state_dict
from engine.deim.hybrid_encoder import RepNCSPELAN4, ConvNormLayer_fuse, SCDown, CSPLayer, TransformerEncoderBlock
from engine.deim.dfine_decoder import DFINETransformer    
from engine.ovdeim.ovdfine_decoder import OVDFINETransformer
from engine.ovdeim.tgfa_ovdfine_decoder import TGFAOVDFINETransformer
from engine.ovdeim.ovdfine_decoder_obb import OVOBBDFINETransformer    
from engine.ovdeim.tgfa_ovdfine_decoder_obb import TGFAOVOBBDFINETransformer
from engine.obbdeim.decoder import OBBDFINETransformer

from engine.extre_module.ultralytics_nn.conv import Concat, Add, Conv
from engine.extre_module.ultralytics_nn.block import Bottleneck, C3_Block, C2f_Block, C3k2_Block, MetaFormer_Block, MetaFormer_Mona, \
                            MetaFormer_SEFN, MetaFormer_Mona_SEFN, NCHW2NLC2NCHW, GetIndexOutput, GetRGBModalityTensor, GetNPYModalityTensor, \
                            MaxSigmoidAttnBlock

RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, LAVENDER, GOLD, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[96m", "\033[95m", "\033[38;5;147m", "\033[38;5;220m", "\033[0m"    
logger = get_logger(__name__)
 
CUSTOM_NN_MODULES = set()
MAMBA_MODULES = set()

__all__ = ['DEIM_MG', 'OVDEIM_MG']     
    
@register(force=True) # 避免因为导入导致的多次注册
class DEIM_MG(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']
    def __init__(self, \
        yaml_path,     
        pretrained=None,
        freeze_stem_only=False,   
        freeze_at=-1,    
        freeze_norm=False, 
        num_classes=80,
        eval_spatial_size=(640, 640) 
    ):
        super().__init__()
        d = yaml_load(yaml_path) 
        backbone, encoder, decoder, self.save = parse_model(d, ch=3, nc=num_classes, eval_spatial_size=eval_spatial_size, verbose=True)    
        self.backbone = backbone     
        self.encoder = encoder
        self.decoder = decoder
     
        # print(self.backbone.state_dict().keys())

        if freeze_at >= 0:     
            self._freeze_parameters(self.backbone[0])
            if not freeze_stem_only:     
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i + 1])   

        if freeze_norm:
            self._freeze_norm(self.backbone)
    
        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"    
            try:
                state = torch.load(pretrained, map_location='cpu') 
                logger.info(f"Loaded stage1 {pretrained} HGNetV2 from local file.")   

                # need_keep_key_prefix_list = ['0.*', '1.*']
                # need_pop_key_list = []
                # for key in state.keys():
                #     need_pop = True 
                #     for keep in need_keep_key_prefix_list:     
                #         if re.match(keep, key):
                #             need_pop = False   
                #             break
                #     if need_pop:
                #         need_pop_key_list.append(key)    
                # for key in need_pop_key_list: 
                #     state.pop(key)

                filtered_state = filter_loadable_state_dict(self.backbone.state_dict(), state)
                logger.info(RED + f'Loading Pretrained State Dict Key Names:{filtered_state.keys()}' + RESET)

                self.backbone.load_state_dict(filtered_state, strict=False)    

            except (Exception, KeyboardInterrupt) as e:
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (not is_dist_available_and_initialized()):   
                    logger.error(f"Loading Backbone Pretrained Weight Error. Message:{str(e)}")
                exit()    
    
    def forward(self, x, targets=None):
        y = []
        for idx, m in enumerate(list(self.backbone.children()) + list(self.encoder.children())):
            # print(idx, m.f, m.i)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers   
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        
        x = self.decoder([y[j] for j in self.decoder.f], targets)
        return x
  
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()     
        return self
    
    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():   
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)    
        return m   

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False


@register(force=True)
class OVDEIM_MG(DEIM_MG):     
    __share__ = ['num_classes', 'eval_spatial_size', 'text_cache_file']
     
    def __init__(
        self,
        yaml_path,
        pretrained=None,  
        freeze_stem_only=False,
        freeze_at=-1,
        freeze_norm=False, 
        num_classes=80,
        eval_spatial_size=(640, 640),
        img_dim=256,
        text_dim=512,
        text_adapter_layers=1,
        text_cache_file: Optional[str] = None,  
    ): 
        super().__init__(
            yaml_path=yaml_path,
            pretrained=pretrained,   
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at, 
            freeze_norm=freeze_norm,    
            num_classes=num_classes,
            eval_spatial_size=eval_spatial_size,
        )   
        self.text_adapter = TextAdapter(text_dim, img_dim, text_adapter_layers)
        self.text_cache_file = text_cache_file
        self.register_buffer("text_feats", torch.empty(0), persistent=False) 
        self.text_prompts: list[str] = []   
        self.text_categories: list[str] = []
        self.prompt_template: Optional[str] = None
        self._forward_text_log_emitted = False
        if text_cache_file:
            self._load_text_cache(text_cache_file)   
   
    def _load_text_cache(self, text_cache_file: str) -> None:     
        text_cache = load_text_cache_payload(torch.load(Path(text_cache_file), map_location="cpu"), text_cache_file)
        self.text_feats = text_cache["text_feats"]
        self.text_prompts = text_cache["prompts"]
        self.text_categories = text_cache["categories"] 
        self.prompt_template = text_cache["prompt_template"]
        logger.info(
            ORANGE + "OV text cache loaded for model: " 
            + summarize_text_cache(text_cache, text_cache_file) + RESET    
        )
    
    def _select_text_feats(self, targets=None) -> torch.Tensor:  
        if targets:   
            per_sample_text_feats = [target.get("text_feats") for target in targets]     
            if all(text_feats is not None for text_feats in per_sample_text_feats):
                return torch.stack(per_sample_text_feats, dim=0)   
        if self.text_feats.numel() == 0:   
            raise RuntimeError("OVDEIM_MG requires text features from targets or text_cache_file.")   
        return self.text_feats    
   
    def forward(self, x, targets=None):     
        y = []

        if isinstance(x, dict):   
            bs = x['rgb'].shape[0]  
        else:     
            bs = x.shape[0]

        text_feats = self.text_adapter(self._select_text_feats(targets))    
        if text_feats.dim() == 2: 
            text_feats = text_feats.unsqueeze(0).repeat(bs, 1, 1)  

        for idx, m in enumerate(list(self.backbone.children()) + list(self.encoder.children())):   
  
            if isinstance(m, MaxSigmoidAttnBlock):
     
                x = m(x, text_feats)
                y.append(x if m.i in self.save else None)
                continue
  
            if m.f != -1: 
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
     
        # if not self._forward_text_log_emitted:   
        #     logger.info(
        #         ORANGE + "OV model text features: "
        #         f"device={text_feats.device}, shape={tuple(text_feats.shape)}, dtype={text_feats.dtype}" + RESET
        #     )
        #     self._forward_text_log_emitted = True

        return self.decoder([y[j] for j in self.decoder.f], targets=targets, text_feats=text_feats)     
    
def yaml_load(file="data.yaml", append_filename=False):  
    """  
    Load YAML data from a file.    
  
    Args:
        file (str, optional): File name. Default is 'data.yaml'.  
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.    
 
    Returns:
        (dict): YAML data and file name.    
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"    
    with open(file, errors="ignore", encoding="utf-8") as f: 
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
     
        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files) 
        if append_filename:
            data["yaml_file"] = str(file)  
        return data 
    
def parse_module(d, i, f, m, args, ch, nc=None, eval_spatial_size=None):
    import ast
    try:
        if m == 'node_mode':
            m = d[m]
            if len(args) > 0:
                if args[0] == 'head_channel':
                    args[0] = int(d[args[0]])
        t = m   
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module  
    except:
        pass

    block_modules = {
        CBIMS, RepHMS, MANet, C3_Block, C2f_Block, C3k2_Block,    
        MetaFormer_Block, MetaFormer_Mona, MetaFormer_SEFN, MetaFormer_Mona_SEFN,
    }    
    selfatt, selfatt_args_index = None, -1   
    if type(args) is list:
        for j, a in enumerate(args): 
            if isinstance(a, str):   
                with contextlib.suppress(ValueError):    
                    try:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)  
                    except:     
                        args[j] = a   
            elif type(a) is dict and m not in block_modules:     
                if 'module' in a: 
                    module_ = a['module']  
                    try:
                        module_ = getattr(torch.nn, module_[3:]) if 'nn.' in module_ else globals()[module_]
                    except Exception as e:
                        raise Exception(f'{module_} is maybe not import in task.py, please check. {e}')
                    module_param = a.get('param', {})
                    for k in module_param:
                        p = module_param[k]     
                        try:
                            module_param[k] = locals()[p] if p in locals() else (getattr(torch.nn, p[3:]) if 'nn.' in p else ast.literal_eval(p))
                        except:
                            module_param[k] = p   
                    args[j] = partial(module_, **module_param)    
                if 'selfatt' in a:
                    selfatt = a['selfatt']    
                    selfatt_args_index = j     
    if selfatt_args_index != -1:   
        args.pop(j)
    
    c2 = ch[-1]     
    if m in {StemBlock, HG_Stage}:
        c1, cmid, c2 = ch[f], args[0], args[1]     
        args = [c1, cmid, c2, *args[2:]]
    elif m is FAENet: 
        c2 = ch[f] 
    elif m in {DPAS, LoGStem, ODALStem, RepStem, SRFD, WGFS}:
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {RepNCSPELAN4, CSPLayer, ConvNormLayer_fuse, SCDown, ESMoE}:   
        c1, c2 = ch[f], args[0]  
        args = [c1, c2, *args[1:]]     
    elif m in {TransformerEncoderBlock}:
        c2 = ch[f]  
        args = [c2, *args]
    elif m in MAMBA_MODULES:
        c2 = ch[f]     
        args = [c2, *args]
    elif m is Concat:     
        c2 = sum(ch[x] for x in f)     
    elif m in {    
        ACAB, CASAB, ContrastDrivenFeatureAggregation, CoordAtt, DeformableLKA, DHPF,
        EMA, FSA, FSDA, GQL, KSFA, LSKBlock, MCA, MLCA, MultiSEAM, SimAM,
        DownsampleConv, ACA
    }: # attention
        c2 = ch[f]  
        args = [c2, *args]
    elif m in {     
        ADHOGSA, AdaptiveSparseSA, BinaryAttention, CBSA, CGTA, SPC_SA, 
        Channel_Transposed_Attention, CascadedGroupAttention, CirculantAttention,
        DAttention, DHOGSA, DPB_Attention, DPWA, DWM_MSA, DilatedGCSA,
        DilatedMWSA, EfficientGlobalSA, FSSA, GSA, LCGA, LRSA, MALA, MSLA,
        MaskUnitAttention, PolaLinearAttention, RSA, Spatial_Frequency_Attention, 
        SAA, SHSA, PatchSA, TAB, Token_Selective_Attention, WDAM,
        BiLevelRoutingAttention_nchw, WCA,   
    }: # transformer  
        c2 = ch[f]
        args = [c2, *args]
    elif m in {
        PSConv, ADown, Conv, A3Conv, DRFD, DMSSP, FSCGD, FSConv, FreqLAWDS, HWD,   
        RouterLAWDS, EdgeLAWDS, SPDConv, V7DownSampling, ContextGuidedBlock_Down, LAWDS,     
        AdaptiveDilatedConv, AGIDWC, CKConv, ConvAttn, Converse2D, DCNv2, DEGConv, DEConv,   
        DeepDiverseBranchBlock, DilatedReparamConv, DiverseBranchBlock, DSA, DySnakeConv, 
        DynamicConv, FDConv, FourierConv, GCConv, gConv, InceptionDWConv2d, MBRConv1,   
        MBRConv3, MBRConv5, MSIDWC, Partial_Conv, PolyConv, RepMBConv, ReparamLargeKernelConv, ScConv,   
        SFS_Conv, SMPConv, WideDiverseBranchBlock, WTConv2d,   
    }: # Conv / downsample
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]] 
    elif m is WaveletPool:
        c2 = ch[f] * 4
    elif m in {CARAFE, DySample, EUCB, EUCB_SC}: # upsample
        c2 = ch[f]
        args = [c2, *args]
    elif m is Converse2D_Up:
        c2 = ch[f]     
        args = [c2, c2, *args]
    elif m is DSUB:
        c1 = ch[f]    
        c2 = c1 // 4
        args = [c1, *args]    
    elif m is WaveletUnPool:   
        c2 = ch[f] // 4
    elif m in CUSTOM_NN_MODULES: # module
        c1, c2 = ch[f], args[0]  
        args = [c1, c2, *args[1:]]
    elif m in {     
        AFFN, ConvolutionalGLU, DFFN, DIFF, DML, DSRG, EDFFN, EFFN,    
        FMFFN, FRFN, KAN, MCCG, SEFN, GLSFFN, MSAFFN, PolyMLP
    }: # mlp
        c1, c2 = ch[f], args[0]
        hidden = args[1] if len(args) > 1 else None
        extra = args[2:] if len(args) > 2 else []
        args = [c1, hidden, c2, *extra]    
    elif m in block_modules: # Block  
        c1, c2 = ch[f], args[0]
        selfatt, selfatt_args_index = None, -1   
        if type(args) is list:    
            parse_scope_locals = locals().copy()
 
            def _resolve_module_from_name(module_name):
                if isinstance(module_name, str):     
                    try: 
                        return getattr(torch.nn, module_name[3:]) if "nn." in module_name else globals()[module_name]
                    except Exception as e:     
                        raise Exception(f"{module_name} is maybe not import in task.py, please check. {e}")
                return module_name  

            def _resolve_module_param_value(value):    
                if isinstance(value, dict):  
                    if "module" in value:     
                        nested_module = _resolve_module_from_name(value["module"])     
                        nested_param = value.get("param", {})    
                        if isinstance(nested_param, dict): 
                            nested_param = {k: _resolve_module_param_value(v) for k, v in nested_param.items()}     
                        else: 
                            nested_param = _resolve_module_param_value(nested_param)     
                        return partial(nested_module, **nested_param) if isinstance(nested_param, dict) else partial(nested_module, nested_param)     
                    return {k: _resolve_module_param_value(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_resolve_module_param_value(v) for v in value]     
                if isinstance(value, tuple):   
                    return tuple(_resolve_module_param_value(v) for v in value)
                if isinstance(value, str):
                    try:     
                        return (     
                            parse_scope_locals[value]
                            if value in parse_scope_locals    
                            else (getattr(torch.nn, value[3:]) if "nn." in value else ast.literal_eval(value))
                        )   
                    except:  
                        return value
                return value
    
            for j, a in enumerate(args):
                if isinstance(a, str):   
                    with contextlib.suppress(ValueError):  
                        try:
                            args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                        except:
                            args[j] = a
                elif type(a) is dict:  
                    if 'module' in a:
                        module_ = _resolve_module_from_name(a['module'])     
                        module_param = a.get('param', {})
                        if isinstance(module_param, dict):
                            module_param = {k: _resolve_module_param_value(v) for k, v in module_param.items()}  
                        else:
                            module_param = _resolve_module_param_value(module_param)
                        args[j] = partial(module_, **module_param) if isinstance(module_param, dict) else partial(module_, module_param)
                    if 'selfatt' in a:     
                        selfatt = a['selfatt']
                        selfatt_args_index = j
        if selfatt_args_index != -1:
            args.pop(j)
        if selfatt is not None:     
            args = ([c1, c2, *args[1:]], {'selfatt':selfatt})
        else:  
            args = [c1, c2, *args[1:]]
    elif m is NCHW2NLC2NCHW:  
        c2 = ch[f]    
        args = [c2, *args]   
    elif m in {HyperComputeModule}:
        c2 = ch[f]     
        args = [c2, *args]  
    elif m is HyperACE:
        c1 = ch[f[1]]
        c2 = args[0]
        args = [c1, c2, *args[1:]]    
    elif m is FullPAD_Tunnel:
        c2 = ch[f[0]] 
    elif m in {  
        CIDAF, WDAF, MFM, MultiScalePCA, MultiScalePCA_Down, MSCRM, ERM, DCGRM, FAAFusion,
        MultiScaleGatedAttn, ContextGuideFusionModule, MFPM, LowFrequencyFeatureFusion,
        HFFE, CAFM, DynamicAlignFusion, GDSAFusion, LCA, HAFFormer, WFU, DPCF, SFSFusion,
        LPRMAlignUpModule, MAFusion,
    }: # featurefusion  
        c1 = [ch[i] for i in f]    
        c2 = args[0]  
        args = [c1, c2, *args[1:]]
    elif m is PST:
        c1, c_up, c2 = ch[f[0]], ch[f[1]], args[0]
        args = [c1, c_up, c2, *args[1:]]  
    elif m is CSFCN:   
        c1 = [ch[i] for i in f]     
        c2 = [args[0], args[0]]  
        args = [c1, args[0], *args[1:]]
    elif m in {MSAM_P4, MSAM_P5}:
        c1 = [ch[i] for i in f]     
        c2 = args[0]
        args = [c1, c2, *args[1:]]
    elif m in {FocusFeature, DynamicFrequencyFocusFeature, AlignmentGuidedFocusFeature}:
        c1 = [ch[i] for i in f]
        c2 = c1[1]   
        args = [c1, *args]  
    # --------------GOLD-YOLO-------------- 
    elif m in {SimFusion_4in, AdvPoolFusion}:
        c2 = sum(ch[x] for x in f) 
    elif m is SimFusion_3in:    
        c1 = [ch[x] for x in f]   
        c2 = args[0]    
        args = [c1, c2]   
    elif m in {IFM, TopBasicLayer}:
        c1 = ch[f]
        c2 = sum(args[0])
        args = [c1, *args]
    elif m in {InjectionMultiSum_Auto_pool, InjectionMultiSum_Auto_pool_Fourier}:
        c1 = ch[f[0]]
        c2 = args[0]    
        args = [c1, c2, *args[1:]]   
    elif m is PyramidPoolAgg:
        c1 = sum(ch[x] for x in f)
        c2 = args[0]     
        args = [c1, c2, *args[1:]]  
    # --------------GOLD-YOLO-------------- 
    elif m is Zoom_cat:     
        c2 = sum(ch[x] for x in f)   
    elif m is ScalSeq:    
        c1 = [ch[x] for x in f]   
        c2 = args[0]    
        args = [c1, c2]
    elif m is Add:
        c2 = ch[f[-1]]  
    elif m is asf_attention_model: 
        c2 = ch[f[0]]
        args = [c2, *args]    
    elif m is HFP: 
        c2 = ch[f] 
        args = [c2, *args]     
    elif m is SDP: 
        c1 = [ch[x] for x in f]
        c2 = args[0]
        args = [c1, c2, *args[1:]]
    elif m in {A3Fusion2, A3Fusion3}:   
        c1 = [ch[x] for x in f] 
        level = args[0]  
        c2 = c1[level]
        args = [level, c1, *args[1:]]
    elif m is A3FPNTri:   
        c1 = [ch[x] for x in f]
        c2 = [args[0], args[0], args[0]]
        args = [c1, args[0], *args[1:]]   
    elif m is ChannelTransformer: 
        c1 = [ch[x] for x in f] 
        c2 = c1 + ([None] if len(c1) == 3 else [])     
        img_size = eval_spatial_size[0] if eval_spatial_size is not None else 640
        args = [c1, img_size, *args] 
    elif m is MSBlock_GQL:
        c1 = ch[f[0]] 
        c2 = args[0]    
        args = [c1, c2, *args[1:]]     
    elif m is HG_Stage_MSBlock:
        c1, cmid, c2 = ch[f[0]], args[0], args[1]
        args = [c1, cmid, c2, *args[2:]] 
    elif m in {DynamicERF, DynamicTanh}:
        c2 = ch[f[0] if isinstance(f, list) else f]
        args = [c2, *args]
    elif m in {GetRGBModalityTensor, GetNPYModalityTensor}:
        if m is GetRGBModalityTensor:  
            c2 = 3     
        else:
            c2 = 1
    elif m in {GetIndexOutput}:
        c2 = ch[f][args[0]]   
    elif m in {  
        DFINETransformer,
        LiteDFINETransformer, 
        DQDFINETransformer,    
        DQSDFINETransformer, 
        DFINESegTransformer, 
        DQSDFINESegTransformer,
        OVDFINETransformer,
        TGFAOVDFINETransformer,
        OVDFINESegTransformer,
        OVOBBDFINETransformer,    
        OBBDFINETransformer,
    }:   
        args["feat_channels"] = [ch[x] for x in f]   
        args["num_classes"] = nc    
        args["eval_spatial_size"] = eval_spatial_size
    elif m is HGNetv2: 
        t = str(m)[8:-2].replace('__main__.', '')  
        m = m(**args) if type(args) is dict else m(*args)
        c2 = [m._out_channels[idx] for idx in m.return_idx]
    elif m in {ConvNeXt_Tiny, ConvNeXt_Small, ConvNeXt_Base, ConvNeXt_Large}:  
        t = str(m)[8:-2].replace('__main__.', '')  # module type     
        m = m(**args)
        c2 = m.dims     
    elif isinstance(m, str):
        t = m   
        # if len(args) == 2:        
        #     m = timm.create_model(m, pretrained=args[0], pretrained_cfg_overlay={'file':args[1]}, features_only=True)   
        # elif len(args) == 1:     
        pretrained = args[0] if isinstance(args, list) and len(args) > 0 else False 
        m = timm.create_model(m, pretrained=pretrained, features_only=True)    
        c2 = m.feature_info.channels()  
    else:
        c2 = ch[f]
 
    if isinstance(m, type) == False and type(m) is not str:     
        m_ = m 
    else:
        if type(args) is dict:  
            m_ = m(**args)
        elif type(args) is list:
            m_ = m(*args)  # module   
        else:     
            m_ = m(*args[0], **args[1])    
        t = str(m)[8:-2].replace('__main__.', '')  # module type
    m_.np = sum(x.numel() for x in m_.parameters())  # number params    
    m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type

    return m_, c2, t, args    

def parse_inference_module(index, source, module_name, args, channels, num_classes=None, eval_spatial_size=None):
    """Build one of the modules used by the released inference graphs."""
    module = getattr(torch.nn, module_name[3:]) if module_name.startswith("nn.") else globals()[module_name]
    if isinstance(args, list):
        args = [None if value == "None" else value for value in args]
    output_channels = channels[-1]

    if module in {StemBlock, HG_Stage}:
        input_channels, middle_channels, output_channels = channels[source], args[0], args[1]
        args = [input_channels, middle_channels, output_channels, *args[2:]]
    elif module in {RepNCSPELAN4, CSPLayer, ConvNormLayer_fuse, SCDown}:
        input_channels, output_channels = channels[source], args[0]
        args = [input_channels, output_channels, *args[1:]]
    elif module is TransformerEncoderBlock:
        output_channels = channels[source]
        args = [output_channels, *args]
    elif module is Concat:
        output_channels = sum(channels[item] for item in source)
    elif module in {GetRGBModalityTensor, GetNPYModalityTensor}:
        output_channels = 3 if module is GetRGBModalityTensor else 1
    elif module is MaxSigmoidAttnBlock:
        input_channels, output_channels = channels[source], args[0]
        args = [input_channels, output_channels, *args[1:]]
    elif module in {TGFAOVDFINETransformer, TGFAOVOBBDFINETransformer}:
        args["feat_channels"] = [channels[item] for item in source]
        args["num_classes"] = num_classes
        args["eval_spatial_size"] = eval_spatial_size

    instance = module(**args) if isinstance(args, dict) else module(*args)
    instance.np = sum(parameter.numel() for parameter in instance.parameters())
    instance.i, instance.f = index, source
    instance.type = str(module)[8:-2].replace("__main__.", "")
    return instance, output_channels, instance.type, args


def parse_model(d, ch, nc, eval_spatial_size, verbose=True):   
    if verbose:     
        logger.info(ORANGE + f"{'':>3}{'from':>10}{'params':>10}  {'module':<60}{'arguments':<30}" + RESET)
    layer_index, ch = 0, [ch]
    backbone_layers, encoder_layers, decoder_model, save, c2 = [], [], None, [], ch[-1]  # layers, savelist, ch out
  
    if verbose:
        logger.info(BLUE + "-"*40 + "BackBone" + "-"*40 + RESET)     
    for f, m, args in d["backbone"]:
   
        m_, c2, t, args = parse_inference_module(
            layer_index, f, m, args, ch, eval_spatial_size=eval_spatial_size
        )

        if verbose:
            logger.info(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
    
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        backbone_layers.append(m_)     
        if layer_index == 0:    
            ch = []     
        ch.append(c2)
  
        layer_index += 1
    
    if verbose:     
        logger.info(BLUE + "-"*40 + "Enchoder" + "-"*40 + RESET)    
    for f, m, args in d["encoder"]:
        
        m_, c2, t, args = parse_inference_module(
            layer_index, f, m, args, ch, eval_spatial_size=eval_spatial_size
        )
    
        if verbose:     
            logger.info(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
        
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        encoder_layers.append(m_)
        ch.append(c2)    
   
        layer_index += 1  
    
    if verbose: 
        logger.info(BLUE + "-"*40 + "Decoder" + "-"*40 + RESET)
    for f, m, args in d["decoder"]:    
        
        m_, c2, t, args = parse_inference_module(
            layer_index, f, m, args, ch, nc, eval_spatial_size
        )
 
        if verbose:
            logger.info(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
 
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist   
        decoder_model = m_
        ch.append(c2) 
    
    # print(ch)
    return nn.Sequential(*backbone_layers), nn.Sequential(*encoder_layers), decoder_model, sorted(save)  
