import numpy as np
import torch


def parameter_names(state_dict, patterns=("classifier",)):
    names = []
    for name in state_dict:
        if any(p in name for p in patterns):
            names.append(name)
    if not names:
        raise ValueError("No matching parameters")
    return names


def vector_from_states(before_state, after_state, patterns=("classifier",)):
    parts = []
    for name in parameter_names(before_state, patterns):
        parts.append((after_state[name].cpu() - before_state[name].cpu()).reshape(-1))
    return torch.cat(parts).numpy()


def state_from_vector(before_state, local_state, vector, patterns=("classifier",)):
    output = {k: v.detach().cpu().clone() for k, v in local_state.items()}
    offset = 0
    for name in parameter_names(before_state, patterns):
        n = before_state[name].numel()
        value = torch.as_tensor(vector[offset:offset+n], dtype=before_state[name].dtype)
        output[name] = before_state[name].cpu() + value.reshape(before_state[name].shape)
        offset += n
    if offset != len(vector):
        raise ValueError("Vector length does not match selected parameters")
    return output


def projected_state(before_state, local_state, basis, alpha, patterns=("classifier",)):
    original = vector_from_states(before_state, local_state, patterns)
    u = np.asarray(basis, dtype=np.float64)
    filtered = original - float(alpha) * ((original @ u) @ u.T)
    output = state_from_vector(before_state, local_state, filtered, patterns)
    return output, original, filtered
