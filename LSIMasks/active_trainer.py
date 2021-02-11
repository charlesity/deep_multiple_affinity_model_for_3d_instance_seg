from inferno.trainers.basic import Trainer
from inferno.utils import python_utils as pyu
from inferno.utils import torch_utils as thu




class ActiveTrainer(Trainer):

    def __init__(self, model=None, active_training=True):
        if model is not None:
            super().__init__(model)
            self._active_training = active_training
            self._mellow_learning = False
    @property
    def active_training(self):
        return self._active_training
    
    @active_training.setter
    def active_training(self,  flag):
        self._active_training = flag

    def train_for(self, num_iterations=None, break_callback=None):
        # revert to standard training if active learning not true
        if not self._active_training:
            super().train_for(num_iterations, break_callback)
        else:
            print("Active training starts here")
            ## implement interactive training here
            pass
            # # Switch model to train mode
            self.train_mode()
            # Call callback
            self.callbacks.call(self.callbacks.BEGIN_OF_TRAINING_RUN,
                            num_iterations=num_iterations)
            # iteration_num is a local clock. There's the global self._iteration_count that keeps
            # actual track of the number of iterations - this is updated by the call to
            # self.next_iteration().
            iteration_num = 0
            while True:
                if num_iterations is not None and iteration_num >= num_iterations:
                    self.console.info("Finished {} iterations. Breaking...".format(num_iterations))
                    break
                # Break if break callback asks us to
                if break_callback is not None and break_callback(iteration_num):
                    self.console.info("Breaking on request from callback.")
                    break
                self.console.progress("Training iteration {} (batch {} of epoch {})."
                                      .format(iteration_num, self._batch_count, self._epoch_count))
                # Call callback
                self.callbacks.call(self.callbacks.BEGIN_OF_TRAINING_ITERATION,
                                    iteration_num=iteration_num)
                # No interrupts while computing - a SIGINT could shoot down the driver if
                # done at the wrong time. Not sure if this has something to do with pinned memory
                # here we actively select informative samples
                with pyu.delayed_keyboard_interrupt():
                    # Get batch
                    print("This is the actual training")
                    batch = self.fetch_next_batch('train')
                    # Send to device and wrap as variable
                    batch = self.wrap_batch(batch, from_loader='train')
                    # Separate inputs from targets
                    inputs, target = self.split_batch(batch, from_loader='train')
                    # Apply model, compute loss and backprop
                    prediction, loss = self.apply_model_and_loss(inputs, target, backward=True,
                                                                 mode='train')
                self.callbacks.call(self.callbacks.AFTER_MODEL_AND_LOSS_IS_APPLIED,
                                    prediction=prediction, loss=loss, iteration_num=iteration_num)
                # Compute metric
                if self.metric_is_defined and self.evaluate_metric_now:
                    self._last_metric_evaluated_at_epoch = self._epoch_count
                    # TODO Make unwrap a method for folks to overload
                    error = self.metric(thu.unwrap(prediction, to_cpu=False),
                                        thu.unwrap(target, to_cpu=False))
                    self.update_state('training_error', thu.unwrap(error))
                else:
                    error = None
                # Update state from computation
                self.update_state('training_inputs', thu.unwrap(inputs))
                self.update_state('training_target', thu.unwrap(target))
                self.update_state('training_prediction', thu.unwrap(prediction))
                self.update_state('training_loss', thu.unwrap(loss))
                # Update state from model's state hooks
                self.update_state_from_model_state_hooks()
                if iteration_num % self.backprop_every == 0:
                    # Update parameters
                    self.optimizer.step()
                    # Zero out the grads
                    self.optimizer.zero_grad()
                # Call callback
                self.callbacks.call(self.callbacks.END_OF_TRAINING_ITERATION,
                                    iteration_num=iteration_num)
                # Prepare for next iteration
                self.next_iteration()
                # Break if validating or saving. It's important that the next_iteration() method is
                # called before checking validate_now and save_now - because otherwise, the iteration
                # counter is never updated after the first save and validate, resulting in an infinite
                # save + validate loop.
                if self.validate_now:
                    self.console.info("Breaking to validate.")
                    break
                if self.save_now:
                    self.console.info("Breaking to save.")
                    break
                iteration_num += 1
            #
            self.callbacks.call(self.callbacks.END_OF_TRAINING_RUN, num_iterations=num_iterations)
            return self