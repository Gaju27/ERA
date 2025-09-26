from imports import optim

def get_scheduler(scheduler_type, optimizer, epochs, steps_per_epoch=None, step_size=6, gamma=0.1, max_lr=0.75):
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError('steps_per_epoch must be provided for OneCycleLR')
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=7.5,
            final_div_factor=7.5,
            cycle_momentum=False
        )
    else:
        raise ValueError('Unknown scheduler type')
