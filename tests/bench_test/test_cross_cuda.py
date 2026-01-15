from kineticEQ.analysis.BGK1D.scheme_comparison import run_scheme_comparison_test
from kineticEQ.analysis.BGK1D.plotting.plot_scheme_comparison_result import plot_cross_scheme_results


def test_cross_scheme_comparison():
    out = run_scheme_comparison_test(
        scheme_list=["explicit", "implicit", "holo"],
        scheme_dt_list=[5e-7, 5e-7, 5e-5],
        tau_tilde=5e-6,
    )

    plot_cross_scheme_results(
        out,
        ref_scheme="explicit",
        filename="scheme_compare.png",
        output_dir="./results/compare",
        show_plots=False,
    )
