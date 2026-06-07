import torch

from equidexflow.utils.distributions import get_dist
from equidexflow.models.vn_dgcnn import VNDGCNNEncoder
from equidexflow.models.vn_vector_fields import VNVectorFields
from equidexflow.utils.ode_solvers import get_ode_solver
from equidexflow.models.equi_grasp_flow import EquiDexFlow
from equidexflow.models.equi_dex_flow import EquiDexFlow as EquiDexFlowDex
from equidexflow.models.contact_decoder import ContactDecoder
from equidexflow.models.force_decoder import ForceDecoder, NormalDecoder
from equidexflow.models.hand_q_decoder import HandQDecoder
from equidexflow.models.hand_q_flow_decoder import HandQFlowDecoder


def get_model(cfg_model):
    name = cfg_model.pop('name')
    checkpoint = cfg_model.get('checkpoint', None)

    if name == 'equidexflow':
        model = _get_equidexflow(cfg_model)
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
    
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location='cpu')

        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])

    return model


def _get_equidexflow(cfg):
    p_uncond = cfg.pop('p_uncond')
    guidance = cfg.pop('guidance')

    init_dist = get_dist(cfg.pop('init_dist'))
    encoder = get_net(cfg.pop('encoder'))
    vector_field = get_net(cfg.pop('vector_field'))
    ode_solver = get_ode_solver(cfg.pop('ode_solver'))

    model = EquiDexFlow(p_uncond, guidance, init_dist, encoder, vector_field, ode_solver)

    return model


def get_net(cfg_net):
    name = cfg_net.pop('name')

    if name == 'vn_dgcnn_enc':
        net = _get_vn_dgcnn_enc(cfg_net)
    elif name == 'vn_vf':
        net = _get_vn_vf(cfg_net)
    else:
        raise NotImplementedError(f"Network {name} is not implemented.")
    
    return net


def _get_vn_dgcnn_enc(cfg):
    net = VNDGCNNEncoder(**cfg)

    return net


def _get_vn_vf(cfg):
    net = VNVectorFields(**cfg)

    return net


# ---------------------------------------------------------------------------
# Dexterous grasp model factory
# ---------------------------------------------------------------------------

def get_dex_model(
    p_uncond: float = 0.1,
    guidance: float = 2.0,
    num_ode_steps: int = 10,
    hand_q_decoder_type: str = "deterministic",
    n_coupling_layers: int = 8,
    surface_proj_tau: float = 0.005,
    wrist_frame: str = "base",
    hand: str = "allegro",
    cond_norm: bool = False,
) -> EquiDexFlowDex:
    """Construct a fully-configured EquiDexFlow for dexterous grasp generation.

    Architecture defaults
    ---------------------
    Encoder      : VNDGCNNEncoder(dims=[1,21,21,42,85,170,341], k=40)  ->  (B,341,3)
    Vector field : VNVectorFields(dims=[346,128,64,2], use_bn=False)
                     input dim = C(341) + x_t_cols(4) + lifted_scalar(1) = 346
                     output dim = 2 vectors  ->  view (B,6) SE(3) velocity
    ODE solver   : SE3_RK4_MK(num_steps=num_ode_steps)
    Decoders     : ContactDecoder(341), ForceDecoder(341), HandQDecoder(341*3)
    Init dist    : SO3_uniform x R3_normal

    Parameters
    ----------
    p_uncond      : classifier-free guidance drop probability
    guidance      : guidance scale at sample time
    num_ode_steps : number of ODE integration steps

    Returns
    -------
    EquiDexFlowDex instance (models.equi_dex_flow.EquiDexFlow)
    """
    from equidexflow.utils.distributions import SO3_uniform_R3_normal
    from equidexflow.utils.ode_solvers import SE3_RK4_MK

    if cond_norm:
        raise NotImplementedError(
            "cond_norm=True requires the LEAP-era model code from machine B "
            "(see docs/MACHINE_B_HANDOFF.md). Allegro v1 uses cond_norm=False."
        )
    if hand not in ("allegro", "leap"):
        raise ValueError(f"unknown hand {hand!r}; expected 'allegro' or 'leap'")

    # Encoder: 7-layer VN-DGCNN  ->  global features (B, 341, 3)
    encoder = VNDGCNNEncoder(
        num_neighbors=40,
        dims=[1, 21, 21, 42, 85, 170, 341],
    )

    # Vector field: input C=341 features + 4 x_t rotation rows + 1 lifted scalar
    vector_field = VNVectorFields(
        dims=[346, 128, 64, 2],
        use_bn=False,
    )

    ode_solver = SE3_RK4_MK(num_steps=num_ode_steps)

    contact_decoder = ContactDecoder(in_channels=341)
    force_decoder   = ForceDecoder(in_channels=341)

    if hand_q_decoder_type == "flow":
        hand_q_decoder = HandQFlowDecoder(
            in_dim=341, wrist_dim=12, n_coupling_layers=n_coupling_layers,
        )
    else:
        hand_q_decoder = HandQDecoder(in_dim=341, wrist_dim=12)

    model = EquiDexFlowDex(
        encoder=encoder,
        vector_field=vector_field,
        ode_solver=ode_solver,
        contact_decoder=contact_decoder,
        force_decoder=force_decoder,
        hand_q_decoder=hand_q_decoder,
        normal_decoder=None,
        p_uncond=p_uncond,
        guidance=guidance,
        init_dist=SO3_uniform_R3_normal,
        surface_proj_tau=surface_proj_tau,
        wrist_frame=wrist_frame,
        hand=hand,
    )

    return model
