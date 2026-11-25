"""Backward-compatible entry point; the current manuscript numbers these plots Figs. 4 and 5.

The CLI is unchanged. Outputs are fig4_alpha_response.pdf and fig5_parameter_responses.pdf.
Legacy scan caches must be rebuilt with --recompute to include the GLS variances
and the current CNN checkpoint identity.
"""
from fig4_5_parameter_response import main

if __name__ == "__main__":
    main()
