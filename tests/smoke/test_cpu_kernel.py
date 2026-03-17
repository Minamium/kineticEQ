from kineticEQ import BGK1D, Config, Engine


def test_smoke_cpu_kernel_implicit():
    cfg = Config(
        model="BGK1D1V",
        scheme="implicit",
        backend="cpu_kernel",
        device="cpu",
        dtype="float64",
        use_tqdm="false",
        log_level="info",
        model_cfg=BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=65, nv=33, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-5, T_total=1e-4),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-2),
            scheme_params=BGK1D.implicit.Params(
                picard_iter=32,
                picard_tol=1e-3,
                abs_tol=1e-13,
            ),
        ),
    )
    Engine(cfg, apply_logging_flag=True).run()
