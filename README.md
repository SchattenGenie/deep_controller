# Deep Controller

@run-time deep controller of simulators

```
python train_deep_controller.py --project_name deep_controller --work_space schattengenie
```


## Comarisons of exact pendulum and approximated pendulum with control network

Without control network two solutions diverge rapidly:

![](before.gif)

With control network there is an insignificant difference:

![](after.gif)


# Tuner

```
python train_tuner.py --project_name tuner --work_space bamasa
```