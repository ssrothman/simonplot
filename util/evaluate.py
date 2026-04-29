from simonplot.typing.Protocols import BaseDatasetProtocol, PrebinnedDatasetProtocol, PrebinnedOperationProtocol, PrebinnedVariableProtocol, VariableProtocol, CutProtocol
from typing import Any

from simonplot.variable.PrebinnedVariable import strip_variable
from simonpy.stats_v2 import apply_jacobian, divide_out_profile, normalize_per_block

def evaluate_on_dataset(dataset : PrebinnedDatasetProtocol, var : PrebinnedVariableProtocol, cut : PrebinnedOperationProtocol) -> Any:
    needed_columns = list(set(var.columns + cut.columns))
    dataset.ensure_columns(needed_columns)

    if dataset.is_stack:
        thevar, details = strip_variable(var)
        accu_val, accu_cov = evaluate_on_dataset(dataset._datasets[0], thevar, cut) # pyright: ignore
        wt = dataset._datasets[0]._weight # pyright: ignore
        accu_val *= wt
        accu_cov *= wt*wt

        for d in dataset._datasets[1:]: # pyright: ignore
            val, cov = evaluate_on_dataset(d, thevar, cut)
            wt = d._weight
            accu_val += val * wt
            accu_cov += cov * wt**2

        if 'normalized_blocks' in details: # type: ignore
            # perform block normalization
            axes = details['normalized_blocks'] # type: ignore
            
            binning = cut.resulting_binning(
                dataset._datasets[0].binning # type: ignore
            )

            accu_val, accu_cov = normalize_per_block(
                accu_val, accu_cov,
                binning, axes
            )

        if 'profiled_blocks' in details: # type: ignore
            # perform profile division
            axes = details['profiled_blocks'] # type: ignore
            
            binning = cut.resulting_binning(
                dataset._datasets[0].binning # type: ignore
            )

            accu_val, accu_cov = divide_out_profile(
                accu_val, accu_cov,
                binning, axes
            )
        
        if 'jac_details' in details: # type: ignore
            # perform jacobian transformation
            binning = cut.resulting_binning( # type: ignore
                dataset._datasets[0] # type: ignore
            )
            accu_val, accu_cov = apply_jacobian(
                accu_val, accu_cov,
                binning, details['jac_details'] # type: ignore
            )

        return accu_val, accu_cov
    else:
        return var.evaluate(dataset, cut) # pyright: ignore[reportArgumentType]