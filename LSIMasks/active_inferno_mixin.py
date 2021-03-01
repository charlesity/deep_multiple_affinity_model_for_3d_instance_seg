from LSIMasks.active_trainer import ActiveTrainer
from speedrun.inferno import InfernoMixin
from speedrun.inferno import IncreaseStepCallback

try:
    from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
except ImportError:
    TensorboardLogger = None

class ActiveInfernoMixin(InfernoMixin):

    @property
    def trainer(self):
        """
        Active inferno trainer. Will be constructed on first use.
        """

        # this line will be modified and uncommented when creating the actual Active Trainer library
        # if inferno is None:
        #     raise ModuleNotFoundError("InfernoMixin requires inferno. You can "
        #                               "install it with `pip install in "
        #                               "pip install inferno-pytorch`")
        # Build trainer if it doesn't exist
        if not hasattr(self, '_trainer'):
            # noinspection PyAttributeOutsideInit
            arg_al = self.get('active_training')
            self._trainer = ActiveTrainer(self.model, active_sl=arg_al['active_sl'], mellow_learning=arg_al['mellow_learning']) \
                .save_to_directory(self.experiment_directory)

            # call all defined bind functions
            for fname in dir(self):
                if fname.startswith('inferno_build_'):
                    print (fname)
                    getattr(self, fname)()

            # add callback to increase step counter
            # noinspection PyUnresolvedReferences
            self._trainer.register_callback(IncreaseStepCallback(self))

            self._trainer.to(self.device)

        return self._trainer



