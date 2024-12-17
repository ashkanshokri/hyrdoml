```python
'''
temporal out of sample validation
'''

from hydroml.config.config import Config
from hydroml.workflow.evaluation import train_finetune_evaluate
    
def get_config(name):
    basins_file = '../sample_data/basins.txt'
    with open(basins_file, 'r') as f:
        catchment_ids = f.read().splitlines()

    config = Config(cal={'periods' : [['1991-01-01', '2014-01-01']], 'catchment_ids':catchment_ids[:10]}, 
                    val={'periods' : [['1985-01-01', '1990-01-01']], 'catchment_ids':catchment_ids[:10]}, 
                    name = name,
                    batch_size=32,
                    device='cpu',
                    )    

    config.set_new_version_name()

    return config


def main(name):
    
    config = get_config(name)
    train_finetune_evaluate(config)




```


```python
main('test')
```

    

    Transforming data: calculating transform parameters and saving to P:\work\sho108\hydroml\results\test\241213180403_d6ed\params.yaml


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(


    valid data points per catchment {0: 5381, 1: 2219, 2: 2259, 3: 4812, 4: 2588, 5: 2680, 6: 2564, 7: 1784, 8: 3428, 9: 7711}


    

    Transforming data: loading transform parameters from P:\work\sho108\hydroml\results\test\241213180403_d6ed\params.yaml


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\results\test\241213180403_d6ed exists and is not empty.
    
      | Name              | Type       | Params | Mode 
    ---------------------------------------------------------
    0 | static_embedding  | Linear     | 15     | train
    1 | dynamic_embedding | Linear     | 5      | train
    2 | lstm              | LSTM       | 266 K  | train
    3 | dropout           | Identity   | 0      | train
    4 | head              | Sequential | 2.6 K  | train
    ---------------------------------------------------------
    268 K     Trainable params
    0         Non-trainable params
    268 K     Total params
    1.075     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode


    P:\work\sho108\hydroml\results\test\241213180403_d6ed



    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 32. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.



    Training: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 2. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:384: `ModelCheckpoint(monitor='val_loss')` could not find the monitored key in the returned metrics: ['lr-Adam', 'train_loss', 'epoch', 'step']. HINT: Did you call `log('val_loss', value)` in the `LightningModule`?
    
    Detected KeyboardInterrupt, attempting graceful shutdown ...



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\call.py:47, in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
         46         return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
    ---> 47     return trainer_fn(*args, **kwargs)
         49 except _TunerExitException:


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\trainer.py:574, in Trainer._fit_impl(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
        568 ckpt_path = self._checkpoint_connector._select_ckpt_path(
        569     self.state.fn,
        570     ckpt_path,
        571     model_provided=True,
        572     model_connected=self.lightning_module is not None,
        573 )
    --> 574 self._run(model, ckpt_path=ckpt_path)
        576 assert self.state.stopped


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\trainer.py:981, in Trainer._run(self, model, ckpt_path)
        978 # ----------------------------
        979 # RUN THE TRAINER
        980 # ----------------------------
    --> 981 results = self._run_stage()
        983 # ----------------------------
        984 # POST-Training CLEAN UP
        985 # ----------------------------


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\trainer.py:1025, in Trainer._run_stage(self)
       1024 with torch.autograd.set_detect_anomaly(self._detect_anomaly):
    -> 1025     self.fit_loop.run()
       1026 return None


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:205, in _FitLoop.run(self)
        204 self.on_advance_start()
    --> 205 self.advance()
        206 self.on_advance_end()


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:363, in _FitLoop.advance(self)
        362 assert self._data_fetcher is not None
    --> 363 self.epoch_loop.run(self._data_fetcher)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\training_epoch_loop.py:140, in _TrainingEpochLoop.run(self, data_fetcher)
        139 try:
    --> 140     self.advance(data_fetcher)
        141     self.on_advance_end(data_fetcher)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\training_epoch_loop.py:250, in _TrainingEpochLoop.advance(self, data_fetcher)
        248 if trainer.lightning_module.automatic_optimization:
        249     # in automatic optimization, there can only be one optimizer
    --> 250     batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
        251 else:


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\optimization\automatic.py:190, in _AutomaticOptimization.run(self, optimizer, batch_idx, kwargs)
        185 # ------------------------------
        186 # BACKWARD PASS
        187 # ------------------------------
        188 # gradient update with accumulated gradients
        189 else:
    --> 190     self._optimizer_step(batch_idx, closure)
        192 result = closure.consume_result()


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\optimization\automatic.py:268, in _AutomaticOptimization._optimizer_step(self, batch_idx, train_step_and_backward_closure)
        267 # model hook
    --> 268 call._call_lightning_module_hook(
        269     trainer,
        270     "optimizer_step",
        271     trainer.current_epoch,
        272     batch_idx,
        273     optimizer,
        274     train_step_and_backward_closure,
        275 )
        277 if not should_accumulate:


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\call.py:167, in _call_lightning_module_hook(trainer, hook_name, pl_module, *args, **kwargs)
        166 with trainer.profiler.profile(f"[LightningModule]{pl_module.__class__.__name__}.{hook_name}"):
    --> 167     output = fn(*args, **kwargs)
        169 # restore current_fx when nested context


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\core\module.py:1306, in LightningModule.optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure)
       1282 r"""Override this method to adjust the default way the :class:`~pytorch_lightning.trainer.trainer.Trainer` calls
       1283 the optimizer.
       1284 
       (...)
       1304 
       1305 """
    -> 1306 optimizer.step(closure=optimizer_closure)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\core\optimizer.py:153, in LightningOptimizer.step(self, closure, **kwargs)
        152 assert self._strategy is not None
    --> 153 step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
        155 self._on_after_step()


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\strategies\strategy.py:238, in Strategy.optimizer_step(self, optimizer, closure, model, **kwargs)
        237 assert isinstance(model, pl.LightningModule)
    --> 238 return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\plugins\precision\precision.py:122, in Precision.optimizer_step(self, optimizer, model, closure, **kwargs)
        121 closure = partial(self._wrap_closure, model, optimizer, closure)
    --> 122 return optimizer.step(closure=closure, **kwargs)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\optim\lr_scheduler.py:137, in LRScheduler.__init__.<locals>.patch_track_step_called.<locals>.wrap_step.<locals>.wrapper(*args, **kwargs)
        136 opt._opt_called = True  # type: ignore[union-attr]
    --> 137 return func.__get__(opt, opt.__class__)(*args, **kwargs)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\optim\optimizer.py:487, in Optimizer.profile_hook_step.<locals>.wrapper(*args, **kwargs)
        483             raise RuntimeError(
        484                 f"{func} must return None or a tuple of (new_args, new_kwargs), but got {result}."
        485             )
    --> 487 out = func(*args, **kwargs)
        488 self._optimizer_step_code()


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\optim\optimizer.py:91, in _use_grad_for_differentiable.<locals>._use_grad(self, *args, **kwargs)
         90     torch._dynamo.graph_break()
    ---> 91     ret = func(self, *args, **kwargs)
         92 finally:


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\optim\adam.py:202, in Adam.step(self, closure)
        201     with torch.enable_grad():
    --> 202         loss = closure()
        204 for group in self.param_groups:


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\plugins\precision\precision.py:108, in Precision._wrap_closure(self, model, optimizer, closure)
        101 """This double-closure allows makes sure the ``closure`` is executed before the ``on_before_optimizer_step``
        102 hook is called.
        103 
       (...)
        106 
        107 """
    --> 108 closure_result = closure()
        109 self._after_closure(model, optimizer)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\optimization\automatic.py:144, in Closure.__call__(self, *args, **kwargs)
        142 @override
        143 def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
    --> 144     self._result = self.closure(*args, **kwargs)
        145     return self._result.loss


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\_contextlib.py:116, in context_decorator.<locals>.decorate_context(*args, **kwargs)
        115 with ctx_factory():
    --> 116     return func(*args, **kwargs)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\optimization\automatic.py:129, in Closure.closure(self, *args, **kwargs)
        126 @override
        127 @torch.enable_grad()
        128 def closure(self, *args: Any, **kwargs: Any) -> ClosureResult:
    --> 129     step_output = self._step_fn()
        131     if step_output.closure_loss is None:


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\optimization\automatic.py:317, in _AutomaticOptimization._training_step(self, kwargs)
        315 trainer = self.trainer
    --> 317 training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
        318 self.trainer.strategy.post_training_step()  # unused hook - call anyway for backward compatibility


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\call.py:319, in _call_strategy_hook(trainer, hook_name, *args, **kwargs)
        318 with trainer.profiler.profile(f"[Strategy]{trainer.strategy.__class__.__name__}.{hook_name}"):
    --> 319     output = fn(*args, **kwargs)
        321 # restore current_fx when nested context


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\strategies\strategy.py:390, in Strategy.training_step(self, *args, **kwargs)
        389     return self._forward_redirection(self.model, self.lightning_module, "training_step", *args, **kwargs)
    --> 390 return self.lightning_module.training_step(*args, **kwargs)


    File \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\hydroml\models\lstm.py:196, in HydroLSTM.training_step(self, batch, batch_idx)
        187 """Training step for the model.
        188 
        189 Args:
       (...)
        194     Dict[str, torch.Tensor]: Dictionary containing the loss.
        195 """
    --> 196 loss = self.loss(batch, batch_idx)
        197 self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    File \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\hydroml\models\lstm.py:180, in HydroLSTM.loss(self, batch, batch_idx)
        171 """Calculate the loss for the given batch.
        172 
        173 Args:
       (...)
        178     torch.Tensor: Computed loss.
        179 """
    --> 180 prediction = self(batch['x_dynamic'], batch['x_static'])
        181 ground_truth = batch['y']


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\nn\modules\module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
       1735 else:
    -> 1736     return self._call_impl(*args, **kwargs)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\nn\modules\module.py:1747, in Module._call_impl(self, *args, **kwargs)
       1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1745         or _global_backward_pre_hooks or _global_backward_hooks
       1746         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1747     return forward_call(*args, **kwargs)
       1749 result = None


    File \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\hydroml\models\lstm.py:161, in HydroLSTM.forward(self, x_dynamic, x_static)
        160 # Pass to LSTM
    --> 161 x, (h, c) = self.lstm(x)
        162 x = self.dropout(x)  # Dropout after LSTM


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\nn\modules\module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
       1735 else:
    -> 1736     return self._call_impl(*args, **kwargs)


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\nn\modules\module.py:1747, in Module._call_impl(self, *args, **kwargs)
       1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1745         or _global_backward_pre_hooks or _global_backward_hooks
       1746         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1747     return forward_call(*args, **kwargs)
       1749 result = None


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\nn\modules\rnn.py:1123, in LSTM.forward(self, input, hx)
       1122 if batch_sizes is None:
    -> 1123     result = _VF.lstm(
       1124         input,
       1125         hx,
       1126         self._flat_weights,
       1127         self.bias,
       1128         self.num_layers,
       1129         self.dropout,
       1130         self.training,
       1131         self.bidirectional,
       1132         self.batch_first,
       1133     )
       1134 else:


    KeyboardInterrupt: 

    
    During handling of the above exception, another exception occurred:


    NameError                                 Traceback (most recent call last)

    Cell In[5], line 1
    ----> 1 main('test')


    Cell In[4], line 28, in main(name)
         25 def main(name):
         27     config = get_config(name)
    ---> 28     train_finetune_evaluate(config)


    File \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\hydroml\workflow\evaluation.py:32, in train_finetune_evaluate(config, save_results)
         31 def train_finetune_evaluate(config: Config, save_results: bool = True):
    ---> 32     exp_base_path = train_evaluate(config, save_results=save_results)
         34     metrics_list = []
         35     for catchment_id in config.get('val')['catchment_ids']:


    File \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\hydroml\workflow\evaluation.py:21, in train_evaluate(config, save_results)
         18 def train_evaluate(config: Config, save_results: bool = True):
         19 
         20     # train the model
    ---> 21     current_path , version_pre = train(config)
         23     model_path = Path(current_path) / version_pre
         24     _ = evaluate_model(model_path, check_point='last', save_results=save_results)


    File \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\hydroml\training\train.py:33, in train(config, cal_split_name, val_split_name)
         29 # make trainer
         31 trainer = get_trainer(config)
    ---> 33 trainer.fit(model, cal_dataloader, val_dataloader)
         35 return config.current_path, config.version


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\trainer.py:538, in Trainer.fit(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
        536 self.state.status = TrainerStatus.RUNNING
        537 self.training = True
    --> 538 call._call_and_handle_interrupt(
        539     self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        540 )


    File c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\call.py:64, in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
         62     if isinstance(launcher, _SubprocessScriptLauncher):
         63         launcher.kill(_get_sigkill_signal())
    ---> 64     exit(1)
         66 except BaseException as exception:
         67     _interrupt(trainer, exception)


    NameError: name 'exit' is not defined



```python

```
