from contextlib import contextmanager

import numpy as np

import src.federated as fed
from src.defense.state_update import projected_state


@contextmanager
def predefined_clients(client_data):
    original = fed.build_client_partitions

    def builder(*args, **kwargs):
        return client_data

    fed.build_client_partitions = builder
    try:
        yield
    finally:
        fed.build_client_partitions = original


@contextmanager
def update_projection(basis, alpha, patterns=("classifier",)):
    original = fed.local_train
    u = np.asarray(basis, dtype=np.float64)

    def defended_local_train(global_model, client_samples, tokenizer, args, mode):
        before = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
        }
        local_state, _, loss, acc = original(
            global_model=global_model,
            client_samples=client_samples,
            tokenizer=tokenizer,
            args=args,
            mode=mode,
        )
        new_state, _, filtered = projected_state(
            before,
            local_state,
            u,
            alpha,
            patterns,
        )
        return new_state, filtered, loss, acc

    fed.local_train = defended_local_train
    try:
        yield
    finally:
        fed.local_train = original
