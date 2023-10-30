import torch
import numpy as np
from tqdm.auto import tqdm


def no_overlap_occlusion(model, spectrum, baseline, window_size, offset):
    with torch.no_grad():
        assert spectrum.shape == baseline.shape
        assert spectrum.ndim == 1

        ref_output = model(
            baseline.unsqueeze(0)
        ).squeeze()  # reference output to compare to
        grad = torch.zeros_like(spectrum, dtype=spectrum.dtype)  # store grads
        new = torch.zeros_like(spectrum, dtype=spectrum.dtype)  # store new array

        offset = offset % window_size
        stride = window_size
        chunks = int(np.ceil(len(spectrum[offset:]) / window_size))
        if offset > 0:
            chunks += 1

        ctr = 0
        mask = torch.ones_like(spectrum, dtype=bool)  # all like spectrum

        for i in range(chunks):
            if offset > 0:
                if i == 0:  # check for offset in first block
                    mask[
                        :offset
                    ] = False  # just mask the first few elements for first chunk
                else:
                    mask[ctr : ctr + window_size] = False
            else:
                mask[
                    ctr : ctr + window_size
                ] = False  # do things normally, mask whole window

            new[:] = torch.where(mask, baseline, spectrum)  # , baseline)
            new_output = model(new.unsqueeze(0)).squeeze()  # calculate new output
            grad[ctr : ctr + window_size] = (ref_output - new_output) / (
                ~mask
            ).sum()  # save difference in all masked pixels

            # put spectral elements back
            if offset > 0:
                if i == 0:  # check for offset in first block
                    mask[
                        :offset
                    ] = True  # just mask the first few elements for first chunk
                    ctr += offset
                else:
                    mask[ctr : ctr + window_size] = True
                    ctr += stride
            else:
                mask[ctr : ctr + window_size] = True
                ctr += stride  # move over by stride

        return grad


def inverse_multiscale_occlusion(
    model,
    spectrum,
    baselines,
    z,
    window_sizes=[8, 16, 32, 64, 128, 256],
    offsets=[0, 0, 0, 0, 0, 0],
    avg_over_baselines=True,
):
    """
    Inverse multiscale occlusion as described in Shen and Melchior (2023).

    Inputs:
    =======
    model: outlier model, which takes in some inputs and returns a scalar output indicating the logP of the input
    spectrum: the spectrum to be explained
    baselines: the baselines to use
    z: redshift of the input spectrum
    window_sizes: the window sizes (and strides) to use
    offsets: the offsets to use for the window sizes
    avg_over_baselines: whether to average over the baselines

    Outputs:
    ========
    attributions: the attributions for each baseline and window size
    aligned_baselines: the aligned baselines
    attribution_stds: the standard deviation of the attributions across the baselines (which can be used to weight the attributions)
    """
    assert len(window_sizes) == len(offsets)
    with torch.no_grad():
        attributions = []
        aligned_baselines = []
        for b in baselines:
            aligned_baselines.append(model.reconstruct(b, z))
        aligned_baselines = torch.stack(aligned_baselines).float()
        for ws, os in tqdm(zip(window_sizes, offsets), total=len(window_sizes)):
            if avg_over_baselines:
                attributions.append(
                    torch.mean(
                        torch.stack(
                            [
                                no_overlap_occlusion(model, spectrum, b, ws, os)
                                for b in aligned_baselines
                            ]
                        ),
                        dim=0,
                    )
                )
            else:
                attributions.append(
                    torch.stack(
                        [
                            no_overlap_occlusion(model, spectrum, b, ws, os)
                            for b in aligned_baselines
                        ]
                    )
                )

        stacked_attributions = torch.stack(attributions).detach().cpu().numpy()
        attribution_stds = stacked_attributions.std(axis=1)
        return (
            stacked_attributions,
            aligned_baselines.detach().cpu().numpy(),
            attribution_stds,
        )
