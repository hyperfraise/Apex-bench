# Apex-error

This script exhibits the problem of apex with 2080ti hardware.

The config of the first machine, `hex` is :
```
2080 Ti
AMD Ryzen Threadripper 1900X 
```
The config of the first machine, `alibaba` is :
```
1080 Ti
Intel(R) Core(TM) i7-6850K
```
It's not due to any kind of single threaded performance (which is bullcrap anyway), because :
```
Single thread perf of 1900x : 112
Single thread perf of i7-6850K : 116
```

You can't convince me that such a difference will have such an effect on deep learning, let alone ANY. But just to be sure, I also switched the cards, and it's the same result.

On both machines :
```
>>> torch.__version__
'1.1.0'
Driver Version: 418.67       CUDA Version: 10.1 
```


## 2080 ti bench

Using O1 mode with apex allows a 2080 to see ~3 samples per second in training with the 3D resnext-101 architecture.

```
veesion@hex:~/Apex-error$ python main.py -bs 8 -bt -hm 2
/home/veesion/Apex-error/resnext.py:121: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
  m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
Selected optimization level O1:  Insert automatic casts around Pytorch functions and Tensor methods.

Defaults for this optimization level are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
Processing user overrides (additional kwargs that are not None)...
After processing overrides, optimization options are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 32768.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 16384.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 8192.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 2048.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 1024.0
306 ms per sample 3.264227441837997 samples per second
325 ms per sample 3.0742666242831227 samples per second
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 512.0
327 ms per sample 3.051127662860935 samples per second
327 ms per sample 3.05291581960959 samples per second
```


Whereas using a "dumb .`half()`" everywhere allows a 2080 to see ~5.7 samples per second.

```
veesion@hex:~/Apex-error$ python main.py -bs 8 -bt -hm 1
/home/veesion/Apex-error/resnext.py:121: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
  m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
166 ms per sample 6.013995896223586 samples per second
175 ms per sample 5.70842944727058 samples per second
173 ms per sample 5.756464869950157 samples per second
173 ms per sample 5.755626853007018 samples per second
```

## 1080 ti bench


```
veesion@alibaba:~/Apex-error$ python main.py -bs 8 -bt -hm 2
/home/veesion/Apex-error/resnext.py:121: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
  m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
Selected optimization level O1:  Insert automatic casts around Pytorch functions and Tensor methods.

Defaults for this optimization level are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
Processing user overrides (additional kwargs that are not None)...
After processing overrides, optimization options are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 32768.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 16384.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 8192.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
^B%wGradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 2048.0
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 1024.0
197 ms per sample 5.063492770789672 samples per second
209 ms per sample 4.763376626968019 samples per second
211 ms per sample 4.719317912208174 samples per second
212 ms per sample 4.710072016963902 samples per second
```

Which is relevant with the "dumb" approach.

```
veesion@alibaba:~/Apex-error$ python main.py -bs 8 -bt -hm 1
/home/veesion/Apex-error/resnext.py:121: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
  m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
197 ms per sample 5.067984641881277 samples per second
207 ms per sample 4.8256311954790085 samples per second
207 ms per sample 4.827193821938975 samples per second
207 ms per sample 4.826493225843475 samples per second
```

With Apex half precision inference, 2080 ti ends up being slower than 1080 ti !
