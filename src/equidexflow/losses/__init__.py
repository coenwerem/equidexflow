from equidexflow.losses.mse_loss import MSELoss
from equidexflow.losses.contact_loss import contact_position_loss, contact_coverage_loss
from equidexflow.losses.force_loss import force_regression_loss, force_direction_loss
from equidexflow.losses.physics_loss import physics_loss


def get_losses(cfg_losses):
    losses = {}

    for cfg_loss in cfg_losses:
        name = cfg_loss.name

        losses[name] = get_loss(cfg_loss)

    return losses


def get_loss(cfg_loss):
    name = cfg_loss.pop('name')

    if name == 'mse':
        loss = MSELoss(**cfg_loss)
    else:
        raise NotImplementedError(f"Loss {name} is not implemented.")

    return loss


def get_physics_losses(cfg=None):
    """Factory: returns a dict of physics loss callables.

    The 'physics' callable matches the signature of ``physics_loss``.
    """
    return {"physics": physics_loss}


def get_contact_losses(cfg=None):
    """Factory: returns a dict of contact loss callables."""
    return {
        "contact_position": contact_position_loss,
        "contact_coverage": contact_coverage_loss,
    }


def get_force_losses(cfg=None):
    """Factory: returns a dict of force loss callables."""
    return {
        "force_regression": force_regression_loss,
        "force_direction":  force_direction_loss,
    }
